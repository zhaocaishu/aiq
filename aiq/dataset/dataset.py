from collections import defaultdict
from typing import List, Dict, Sequence, Tuple

import torch
import numpy as np
import pandas as pd

from aiq.dataset.loader import DataLoader
from aiq.utils.functional import (
    fillna,
    zscore,
)


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
        split: str = "train",
    ):
        start_time, end_time = segments[split]
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

    def get_training_data(self, feature_names=None, label_names=None):
        feature_names = self.feature_names if feature_names is None else feature_names
        label_names = self.label_names if label_names is None else label_names
        return self.data[feature_names].to_numpy(), self.data[label_names].to_numpy()


class CSDataset(Dataset):
    """Tabular inputs matching PPNet's daily CS features and eligible stocks."""

    def __init__(
        self,
        data: pd.DataFrame,
        segments: Dict[str, Tuple[str, str]],
        universe: str,
        seq_len: int,
        data_dir: str = "",
        feature_names: Sequence[str] = (),
        label_names: Sequence[str] = (),
        split: str = "train",
    ):
        self.feature_names = [name for name in feature_names if name.startswith("CS_")]
        self.label_names = list(label_names)
        self.data_dir = data_dir
        if not self.feature_names or seq_len < 1:
            raise ValueError("CSDataset requires CS_ features and a positive seq_len.")

        start, end = segments[split]
        if not data.index.is_monotonic_increasing:
            data = data.sort_index()
        # Count history before slicing the split or filtering daily membership.
        history = data.groupby(level="Instrument", sort=False).cumcount().loc[start:end]
        window = data.loc[start:end]
        members = DataLoader.load_instruments(data_dir, universe, start, end)
        member_index = pd.MultiIndex.from_frame(members[["Date", "Instrument"]])
        eligible = (history >= seq_len - 1) & window.index.isin(member_index)
        self.data = window.loc[eligible, self.feature_names + self.label_names].copy()

        cs = slice(0, len(self.feature_names))
        for rows in self.data.groupby(level="Date", sort=False).indices.values():
            features = self.data.iloc[rows, cs].to_numpy()
            self.data.iloc[rows, cs] = fillna(zscore(features, clip_outlier=True))

    def get_training_data(self, feature_names=None, label_names=None):
        """Standardize targets for fitting while keeping raw returns for evaluation."""
        feature_names = self.feature_names if feature_names is None else feature_names
        label_names = self.label_names if label_names is None else label_names
        if not set(feature_names).issubset(self.feature_names):
            raise ValueError("CSDataset only supports its configured CS features.")
        raw = self.data[label_names].to_numpy()
        labels = np.empty_like(raw, dtype=np.result_type(raw.dtype, np.float32))
        for rows in self.data.groupby(level="Date", sort=False).indices.values():
            labels[rows] = zscore(raw[rows])
        valid = np.isfinite(labels).all(axis=1)
        if not valid.any():
            raise ValueError("No finite training labels in this split.")
        return self.data.loc[valid, feature_names].to_numpy(), labels[valid]


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
        split: str = "train",
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
            split (str, optional): Dataset split ('train', 'val', 'test'). Defaults to "train".
        """
        self.data_dir = data_dir
        self.data = data.copy(deep=False)
        self.seq_len = seq_len
        self.feature_names = feature_names
        self.label_names = label_names
        self.use_augmentation = use_augmentation
        self.split = split

        self.start_time, self.end_time = segments[self.split]

        # Build feature index positions for efficiency
        self._build_feature_indices()

        # Load and set instrument filter if provided
        self._load_instruments(data_dir, universe)

        # Setup the time series data
        self._setup_time_series()

    def _build_feature_indices(self):
        """Build indices for different feature categories to avoid repeated lookups."""
        self.industry_index_l1 = next(
            (i for i, name in enumerate(self.feature_names) if name == "IND_CLS_L1"),
            None,
        )
        self.industry_index_l2 = next(
            (i for i, name in enumerate(self.feature_names) if name == "IND_CLS_L2"),
            None,
        )
        self.stock_ts_feature_indices = [
            i for i, name in enumerate(self.feature_names) if name.startswith("TS_")
        ]
        self.stock_cs_feature_indices = [
            i for i, name in enumerate(self.feature_names) if name.startswith("CS_")
        ]
        self.stock_fund_feature_indices = [
            i for i, name in enumerate(self.feature_names) if name.startswith("FUND_")
        ]
        self.market_feature_indices = [
            i for i, name in enumerate(self.feature_names) if name.startswith("MKT_")
        ]

    def _load_instruments(self, data_dir: str, universe: str):
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
            self.instruments_d = set(zip(df["Instrument"], df["Date"]))
        else:
            self.instruments_d = None

    def _in_segment_and_pool(self, instrument: str, date: str) -> bool:
        """
        Validate whether current sample should be included.
        """
        if date < self.start_time or date > self.end_time:
            return False

        if (
            self.instruments_d is not None
            and (instrument, date) not in self.instruments_d
        ):
            return False

        return True

    def _setup_time_series(self):
        """Prepare the time series data: sort index, extract features/labels, create slices, and group by date."""
        # Ensure index is (Instrument, Date) and sorted
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
        ts_slices = self._create_ts_slices(self._index, self.seq_len)

        # Group valid slices by date
        daily_slices = defaultdict(list)
        for i, (instrument, date) in enumerate(self._index):
            if not self._in_segment_and_pool(instrument, date):
                continue

            # Ensure slice has exact sequence length
            current_slice = ts_slices[i]
            if current_slice.stop - current_slice.start != self.seq_len:
                continue

            daily_slices[date].append(current_slice)

        # Store dates and corresponding slices
        self._daily_dates = sorted(list(daily_slices.keys()))  # Sort for consistency
        self._daily_slices = [daily_slices[date] for date in self._daily_dates]

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
        assert index.nlevels == 2, "Index must have 2 levels (Instrument, Date)"
        assert index.is_monotonic_increasing, "Index must be sorted in increasing order"
        assert seq_len > 0, "Sequence length must be greater than 0"

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
        # Retrieve data slices corresponding to the current query date
        ts_slices = self._daily_slices[index]

        # Extract the terminal index for each slice (last time step)
        sample_indices = np.array([sl.stop - 1 for sl in ts_slices])

        # Aggregate feature sequences into a 3D tensor: [Batch, Seq_Len, Features]
        features = np.stack([self._features[sl] for sl in ts_slices])

        # Append ground truth labels if in training/validation mode
        if self._labels is not None:
            labels = np.array([self._labels[sl.stop - 1] for sl in ts_slices])
        else:
            labels = None

        # Split features into functional subsets
        stock_ts_features = features[:, :, self.stock_ts_feature_indices]
        stock_cs_features = features[:, -1, self.stock_cs_feature_indices]
        stock_fund_features = features[:, -1, self.stock_fund_feature_indices]
        market_state_features = features[:, :, self.market_feature_indices]

        # Data Normalization Pipeline
        # TS scales are prepared by the handler; retain them across dates and pools.
        stock_cs_features = zscore(stock_cs_features, clip_outlier=True)
        stock_fund_features = zscore(stock_fund_features, clip_outlier=True)

        # Impute missing values (NaNs) with zero
        stock_ts_features = fillna(stock_ts_features, fill_value=0.0)
        stock_cs_features = fillna(stock_cs_features, fill_value=0.0)
        stock_fund_features = fillna(stock_fund_features, fill_value=0.0)

        # Construct the finalized data payload for model input
        data_dict = {
            "sample_indices": sample_indices.astype(np.int64),
            "ts_slices": [(s.start, s.stop) for s in ts_slices],
            "stock_industry_ids": np.stack(
                [
                    features[:, -1, self.industry_index_l1].astype(np.int64),
                    features[:, -1, self.industry_index_l2].astype(np.int64),
                ],
                axis=1,
            ),
            "stock_ts_features": stock_ts_features,
            "stock_cs_features": stock_cs_features,
            "stock_fund_features": stock_fund_features,
            "market_state_features": market_state_features,
        }

        if labels is not None:
            data_dict["labels"] = zscore(labels)

        return data_dict

    def __len__(self) -> int:
        """Return the number of dates (batches) in the dataset."""
        return len(self._daily_dates)
