from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
import torch

from aiq.dataset.loader import DataLoader
from aiq.utils.functional import ts_robust_zscore, fillna, drop_extreme_label


class Dataset(torch.utils.data.Dataset):
    """
    Preparing data for model training and inference.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        segments: dict,
        feature_names: List[str] = [],
        label_names: List[str] = [],
        mode: str = "train",
    ):
        start_time, end_time = segments[mode]
        self._data = data.loc[start_time:end_time].copy()
        self._feature_names = feature_names
        self._label_names = label_names

    def __getitem__(self, index):
        return self._data.iloc[[index]]

    def __len__(self):
        return self._data.shape[0]

    @property
    def data(self):
        return self._data

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def label_names(self):
        return self._label_names


class TSDataset(Dataset):
    """
    Time series dataset.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        segments: dict,
        seq_len: int,
        data_dir: str = "",
        universe: str = "",
        feature_names: List[str] = [],
        label_names: List[str] = [],
        norm_feature_start_index: int = 0,
        norm_feature_end_index: int = None,
        use_augmentation: bool = False,
        mode: str = "train",
    ):
        self._data = data.copy(deep=False)
        self.seq_len = seq_len
        self._feature_names = feature_names
        self._label_names = label_names
        self.norm_feature_start_index = norm_feature_start_index
        self.norm_feature_end_index = norm_feature_end_index or len(self._feature_names)
        self.use_augmentation = use_augmentation
        self.mode = mode

        self.start_time, self.end_time = segments[mode]

        if data_dir and universe:
            df = DataLoader.load_instruments(
                data_dir, universe, self.start_time, self.end_time
            )
            self.instruments_set = set(zip(df["Instrument"], df["Date"]))
        else:
            self.instruments_set = None

        self._setup_time_series()

    def _setup_time_series(self):
        self._data.index = self._data.index.swaplevel()
        self._data.sort_index(inplace=True)

        self._feature = self._data[self._feature_names].to_numpy(copy=False)
        self._label = (
            self._data[self._label_names].to_numpy(copy=False)
            if self._label_names
            else None
        )
        self._index = self._data.index

        daily_slices = defaultdict(list)
        data_slices = self._create_ts_slices(self._index, self.seq_len)
        for i, (code, date) in enumerate(self._index):
            # Skip outside the desired time window
            if not (self.start_time <= date <= self.end_time):
                continue

            # If filtering by instruments, skip missing pairs
            if (
                self.instruments_set is not None
                and (code, date) not in self.instruments_set
            ):
                continue

            # Only keep slices with length equal to seq_len
            data_slice = data_slices[i]
            if data_slice.stop - data_slice.start == self.seq_len:
                daily_slices[date].append(data_slice)

        self._daily_dates = list(daily_slices.keys())
        self._daily_slices = list(daily_slices.values())

        daily_summary = {date: len(slices) for date, slices in daily_slices.items()}
        print(f"Mode: {self.mode}. Sampled daily counts:", daily_summary)

    def _create_ts_slices(self, index, seq_len):
        assert isinstance(index, pd.MultiIndex), "unsupported index type"
        assert seq_len > 0, "sequence length should be larger than 0"
        assert index.is_monotonic_increasing, "index should be sorted"
        sample_count_by_insts = index.to_series().groupby(level=0).size().values
        start_index_of_insts = np.roll(np.cumsum(sample_count_by_insts), 1)
        start_index_of_insts[0] = 0
        slices = []
        for cur_loc, cur_cnt in zip(start_index_of_insts, sample_count_by_insts):
            for stop in range(1, cur_cnt + 1):
                end = cur_loc + stop
                start = max(end - seq_len, cur_loc)
                slices.append(slice(start, end))
        slices = np.array(slices, dtype="object")
        assert len(slices) == len(index)
        return slices

    def __getitem__(self, index):
        """Return sample indices, features, and standardized labels (if available) based on the given index."""
        # Time slices for the current date
        slices = self._daily_slices[index]

        # Original index list corresponding to the current date
        indices = np.array([slice.stop - 1 for slice in slices])

        # Extract feature sequences based on each slice and stack into a 3D array [num_samples, time_steps, num_features]
        features = np.stack([self._feature[slice] for slice in slices])

        # Apply Robust Z-score normalization to selected feature columns
        start, end = self.norm_feature_start_index, self.norm_feature_end_index
        features[:, :, start:end] = ts_robust_zscore(
            features[:, :, start:end], clip_outlier=True
        )

        # Fill missing features
        features = fillna(features, fill_value=0.0)

        # If no labels are defined, return indices and features only
        if not self._label_names:
            return indices, features, None

        # Extract labels from the last time step of each sequence
        labels = np.array([self._label[slice.stop - 1] for slice in slices])

        # In training mode, filter out samples with extreme label values
        if self.mode == "train":
            mask, labels = drop_extreme_label(labels)
            indices = indices[mask]
            features = features[mask]

        # Apply cross-sectional rank percentile normalization to labels
        ranks = labels.argsort(axis=0).argsort(axis=0)
        labels = ranks / (labels.shape[0] - 1)
        labels = labels.astype(np.float32)

        return indices, features, labels

    def __len__(self):
        return len(self._daily_dates)
