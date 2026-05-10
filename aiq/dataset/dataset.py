from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set

import torch
import numpy as np
import pandas as pd

from aiq.dataset.loader import DataLoader
from aiq.utils.functional import (
    fillna,
    zscore,
    robust_zscore,
    ts_ohlcv_normalize,
    drop_extreme_label,
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
        self.data_dir = data_dir
        self.data = data.copy(deep=False)
        self.seq_len = seq_len
        self.feature_names = feature_names
        self.label_names = label_names
        self.use_augmentation = use_augmentation
        self.mode = mode

        self.start_time, self.end_time = segments[mode]

        # Build feature index positions for efficiency
        self._build_feature_indices()

        # Load and set instrument filter if provided
        self.instruments_set = self._load_instruments(data_dir, universe)

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

    def _is_valid_sample(self, instrument: str, date: str) -> bool:
        """
        Validate whether current sample should be included.
        """
        if date < self.start_time or date > self.end_time:
            return False

        if (
            self.instruments_set is not None
            and (instrument, date) not in self.instruments_set
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
            if not self._is_valid_sample(instrument, date):
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

            if self.mode == "train":
                mask, filtered_labels = drop_extreme_label(labels)

                labels = filtered_labels
                features = features[mask]
                sample_indices = sample_indices[mask]
                ts_slices = [s for s, m in zip(ts_slices, mask) if m]
        else:
            labels = None

        # Split features into functional subsets
        stock_ts_features = features[:, :, self.stock_ts_feature_indices]
        stock_cs_features = features[:, -1, self.stock_cs_feature_indices]
        stock_fund_features = features[:, -1, self.stock_fund_feature_indices]
        market_state_features = features[:, -1, self.market_feature_indices]

        # Data Normalization Pipeline
        stock_ts_features = zscore(ts_ohlcv_normalize(stock_ts_features))
        stock_cs_features = robust_zscore(stock_cs_features, clip_outlier=True)
        stock_fund_features = robust_zscore(stock_fund_features, clip_outlier=True)

        # Impute missing values (NaNs) with zero
        stock_ts_features = fillna(stock_ts_features, fill_value=0.0)
        stock_cs_features = fillna(stock_cs_features, fill_value=0.0)
        stock_fund_features = fillna(stock_fund_features, fill_value=0.0)

        # Construct the finalized data payload for model input
        data_dict = {
            "sample_indices": sample_indices.astype(np.int64),
            "ts_slices": ts_slices,
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


class MultiscaleTSDataset(TSDataset):
    """
    Multi-scale time series dataset that extends TSDataset by incorporating
    minute-level features (e.g., 5-minute bars) for each sample.

    This class preloads all required intraday data into memory during initialization
    to ensure high performance during training and inference.
    """

    def __init__(
        self,
        *args,
        minute_bar: int = 5,
        minute_feature_names: List[str] = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "AMount",
        ],
        **kwargs,
    ):
        """
        Initialize the multi-scale dataset and preload intraday data.

        Args:
            minute_bar (int, optional): Minute interval for intraday data (e.g., 5 for 5-min bars).
            minute_feature_names (List[str], optional): List of minute-level feature columns.
        """
        super().__init__(*args, **kwargs)

        self.minute_bar = minute_bar
        self.minutes_per_day = int(240 / self.minute_bar)
        self.minute_feature_names = minute_feature_names

        # In-memory storage for preloaded minute data: {instrument: {date: np.ndarray}}
        self._minute_data_store: Dict[str, Dict[pd.Timestamp.date, np.ndarray]] = {}

        # Preload all minute-level data to avoid disk I/O during training
        self._preload_minute_data()

    def _preload_minute_data(self):
        """
        Load, process, and cache all relevant minute-level data into memory.
        This optimizes the training loop by moving I/O and grouping to initialization.
        """
        all_instruments = self.data.index.get_level_values(0).unique()

        for instrument in all_instruments:
            # Load raw minute data for the entire period
            df = DataLoader.load_instrument_features(
                self.data_dir,
                instrument,
                timestamp_col="Trade_time",
                start_time=self.start_time,
                end_time=self.end_time,
                freq=f"{self.minute_bar}min",
            )
            if df.empty:
                continue

            df["Trade_time"] = pd.to_datetime(df["Trade_time"])
            df["Date"] = df["Trade_time"].dt.date

            # Apply adjustment factor to OHLC prices
            if "Adj_factor" in df.columns:
                for col in ["Open", "High", "Low", "Close"]:
                    df[col] = df[col] * df["Adj_factor"]

            # Group by date and convert to NumPy for O(1) access during training
            inst_data = {}
            for d, group in df.groupby("Date"):
                vals = group[self.minute_feature_names].to_numpy(dtype=np.float32)
                # Only store complete intraday sequences
                if len(vals) == self.minutes_per_day:
                    inst_data[d] = vals

            self._minute_data_store[instrument] = inst_data

    def _get_minute_sequence(
        self, instrument: str, dates: List[pd.Timestamp]
    ) -> np.ndarray:
        """
        Construct minute-level sequence from preloaded memory store.

        Args:
            instrument (str): Stock instrument code.
            dates (List[pd.Timestamp]): List of dates corresponding to seq_len.

        Returns:
            np.ndarray: Minute-level feature array with shape [seq_len * 48, D].
        """
        inst_store = self._minute_data_store.get(instrument, {})
        minute_seq = []

        for d in dates:
            d_date = pd.Timestamp(d).date()
            day_data = inst_store.get(d_date)
            # Ensure full intraday coverage; otherwise pad with zeros
            if day_data is None:
                pad = np.zeros(
                    (self.minutes_per_day, len(self.minute_feature_names)),
                    dtype=np.float32,
                )
                minute_seq.append(pad)
            else:
                minute_seq.append(day_data)

        return np.concatenate(minute_seq, axis=0)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """
        Retrieve a batch of multi-scale data for the given index (date-level batch).

        Args:
            index (int): Index of the date in the dataset.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing:
                - original daily features (from TSDataset)
                - minute-level features (multi-scale)
        """
        # Get base daily data (features, labels, sample_indices)
        data_dict = super().__getitem__(index)

        # Sync with filtered samples (handling potential extreme label dropping)
        sample_indices = data_dict["sample_indices"]
        ts_slices = data_dict["ts_slices"]
        minute_features_list = []

        for idx, sl in zip(sample_indices, ts_slices):
            # Retrieve instrument and the sequence of dates for the window
            instrument, date = self._index[idx]

            # Reconstruct the sequence of dates for the current sample
            seq_dates = [self._index[i][1] for i in range(sl.start, sl.stop)]

            # Fetch aligned minute sequences from memory
            minute_seq = self._get_minute_sequence(instrument, seq_dates)
            minute_features_list.append(minute_seq)

        # Aggregate and normalize minute-level features
        minute_features = np.stack(minute_features_list)
        minute_features = zscore(ts_ohlcv_normalize(minute_features))
        minute_features = fillna(minute_features, fill_value=0.0)

        # Add to output dict
        data_dict["stock_intraday_ts_features"] = minute_features.astype(np.float32)

        return data_dict
