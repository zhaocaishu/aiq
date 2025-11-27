from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set

import torch
import numpy as np
import pandas as pd

from aiq.dataset.loader import DataLoader
from aiq.utils.functional import ts_robust_zscore, fillna


class Dataset(torch.utils.data.Dataset):
    """
    Preparing data for model training and inference.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        segments: Dict[str, Tuple[str, str]],
        data_dir: str = "",
        feature_names: List[str] = [],
        label_names: List[str] = [],
        mode: str = "train",
    ):
        start_time, end_time = segments[mode]
        self.data = data.loc[start_time:end_time].copy()
        self.data_dir = data_dir
        self.feature_names = feature_names
        self.label_names = label_names

    def __getitem__(self, index):
        row = self.data.iloc[index]
        data_dict = {"features": row[self.feature_names].to_numpy()}
        if self.label_names:
            data_dict["labels"] = row[self.label_names].to_numpy()
        return data_dict

    def __len__(self):
        return self.data.shape[0]


class TSDataset(Dataset):
    """
    Time series dataset for handling financial or stock data in a PyTorch-compatible format.
    This class processes time series data, applies filters, and prepares sequences for model input.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        segments: Dict[str, Tuple[str, str]],
        data_dir: str = "",
        universe: str = "",
        seq_len: int = 8,
        feature_names: List[str] = [],
        label_names: List[str] = [],
        use_augmentation: bool = False,
        mode: str = "train",
    ):
        """
        Initialize the dataset.

        Args:
            data (pd.DataFrame): The input DataFrame containing time series data.
            segments (Dict[str, Tuple[str, str]]): Dictionary mapping modes (e.g., 'train') to (start_time, end_time) tuples.
            data_dir (str, optional): Directory for loading instrument data. Defaults to "".
            universe (str, optional): Universe identifier for instrument filtering. Defaults to "".
            seq_len (int, optional): Length of each time series sequence. Defaults to 8.
            feature_names (List[str], optional): List of feature column names. Defaults to [].
            label_names (List[str], optional): List of label column names. Defaults to [].
            use_augmentation (bool, optional): Whether to apply data augmentation. Defaults to False.
            mode (str, optional): Dataset mode ('train', 'val', 'test'). Defaults to "train".
        """
        self.data = data.copy(deep=False)
        self.seq_len = seq_len
        self.feature_names = feature_names
        self.label_names = label_names
        self.use_augmentation = use_augmentation
        self.mode = mode

        self.start_time, self.end_time = segments[mode]

        # Precompute feature index positions for efficiency
        self._precompute_feature_indices()

        # Load and set instrument filter if provided
        self.instruments_set = self._load_instruments(data_dir, universe)

        # Setup the time series data
        self._setup_time_series()

    def _precompute_feature_indices(self):
        """Precompute indices for different feature categories to avoid repeated lookups."""
        self.industry_index = next(
            (i for i, name in enumerate(self.feature_names) if name == "IND_CLS"), None
        )
        self.stock_ts_feature_indices = [
            i for i, name in enumerate(self.feature_names) if name.startswith("TS_")
        ]
        self.stock_cs_feature_indices = [
            i for i, name in enumerate(self.feature_names) if name.startswith("CS_")
        ]
        self.market_feature_indices = [
            i for i, name in enumerate(self.feature_names) if name.startswith("MKT_")
        ]

    def _load_instruments(
        self, data_dir: str, universe: str
    ) -> Optional[Set[Tuple[str, str]]]:
        """
        Load instruments from the specified directory and universe, if provided.

        Args:
            data_dir (str): Directory for instrument data.
            universe (str): Universe identifier.

        Returns:
            Optional[Set[Tuple[str, str]]]: Set of (instrument, date) tuples, or None if not provided.
        """
        if data_dir and universe:
            df = DataLoader.load_instruments(
                data_dir, universe, self.start_time, self.end_time
            )
            return set(zip(df["Instrument"], df["Date"]))
        return None

    def _setup_time_series(self):
        """Prepare the time series data: sort index, extract features/labels, create slices, and group by date."""
        # Ensure index is (instrument, date) and sorted
        self.data.index = self.data.index.swaplevel()
        self.data.sort_index(inplace=True)

        # Extract features and labels as NumPy arrays for faster access
        self._features = self.data[self.feature_names].to_numpy(copy=False)
        self._labels = (
            self.data[self.label_names].to_numpy(copy=False)
            if self.label_names
            else None
        )
        self._index = self.data.index

        # Create time series slices for each data point
        slices = self._create_ts_slices(self._index, self.seq_len)

        # Group valid slices by date
        daily_slices = defaultdict(list)
        for i, (code, date) in enumerate(self._index):
            # Filter by time window
            if date < self.start_time or date > self.end_time:
                continue

            # Filter by instruments if set
            if (
                self.instruments_set is not None
                and (code, date) not in self.instruments_set
            ):
                continue

            # Ensure slice has exact sequence length
            current_slice = slices[i]
            if current_slice.stop - current_slice.start != self.seq_len:
                continue

            daily_slices[date].append(current_slice)

        # Store dates and corresponding slices
        self._daily_dates = sorted(list(daily_slices.keys()))  # Sort for consistency
        self._daily_slices = [daily_slices[date] for date in self._daily_dates]

        # Log daily counts for debugging
        daily_counts = {
            date: len(slices)
            for date, slices in zip(self._daily_dates, self._daily_slices)
        }
        print(f"Mode: {self.mode}. Sampled daily counts: {daily_counts}")

    def _create_ts_slices(self, index: pd.MultiIndex, seq_len: int) -> np.ndarray:
        """
        Create sliding window slices for time series data grouped by instrument.

        Args:
            index (pd.MultiIndex): MultiIndex with levels (instrument, date).
            seq_len (int): Length of each sequence.

        Returns:
            np.ndarray: Array of slice objects for each data point.
        """
        assert isinstance(index, pd.MultiIndex), "Unsupported index type"
        assert seq_len > 0, "Sequence length must be greater than 0"
        assert index.is_monotonic_increasing, "Index must be sorted in increasing order"

        # Count samples per instrument
        sample_count_by_insts = index.to_series().groupby(level=0).size().values

        # Compute starting indices for each instrument
        start_index_of_insts = np.roll(np.cumsum(sample_count_by_insts), 1)
        start_index_of_insts[0] = 0

        slices = []
        for cur_loc, cur_cnt in zip(start_index_of_insts, sample_count_by_insts):
            for stop in range(1, cur_cnt + 1):
                end = cur_loc + stop
                start = max(end - seq_len, cur_loc)
                slices.append(slice(start, end))

        return np.array(slices, dtype="object")

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """
        Retrieve a batch of data for the given index (corresponding to a date).

        Args:
            index (int): Index of the date in the dataset.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing sample indices, features, and optionally labels.
        """
        # Get slices for the current date
        slices = self._daily_slices[index]

        # Get the ending indices for each slice (last time step)
        sample_indices = np.array([sl.stop - 1 for sl in slices])

        # Extract feature sequences [num_samples, seq_len, num_features]
        features = np.stack([self._features[sl] for sl in slices])

        # Normalize specific feature subsets
        stock_ts_features = ts_robust_zscore(
            features[:, :, self.stock_ts_feature_indices], clip_outlier=True
        )
        stock_cs_features = ts_robust_zscore(
            features[:, -1:, self.stock_cs_feature_indices], clip_outlier=True
        ).squeeze(1)
        market_features = features[:, -1, self.market_feature_indices]

        # Fill NaNs with 0.0
        stock_ts_features = fillna(stock_ts_features, fill_value=0.0)
        stock_cs_features = fillna(stock_cs_features, fill_value=0.0)
        market_features = fillna(market_features, fill_value=0.0)

        # Construct data dictionary
        data_dict = {
            "sample_indices": sample_indices.astype(np.int64),
            "industry_ids": features[:, -1, self.industry_index].astype(np.int64),
            "stock_ts_features": stock_ts_features,
            "stock_cs_features": stock_cs_features,
            "market_features": market_features,
        }

        # Add labels if available
        if self._labels is not None:
            labels = np.array([self._labels[sl.stop - 1] for sl in slices])
            data_dict["labels"] = labels

        return data_dict

    def __len__(self) -> int:
        """Return the number of dates (batches) in the dataset."""
        return len(self._daily_dates)
