from collections import defaultdict
from typing import List, Dict, Tuple, Optional

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
    Time series dataset.
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
        mode: str = "train",
        use_augmentation: bool = False,
    ):
        self.data = data.copy(deep=False)
        self.seq_len = seq_len
        self.feature_names = feature_names
        self.label_names = label_names
        self.mode = mode
        self.start_time, self.end_time = segments[mode]
        self.use_augmentation = use_augmentation

        # Precompute index positions for features
        self.industry_indices = [
            i for i, name in enumerate(self.feature_names) if name == "IND_CLS"
        ]
        self.stock_feature_indices = [
            i
            for i, name in enumerate(self.feature_names)
            if name.startswith("CS_") or name.startswith("TS_")
        ]
        self.market_feature_indices = [
            i for i, name in enumerate(self.feature_names) if name.startswith("MKT_")
        ]

        # Instrument filter
        self.instruments_set = None
        if data_dir and universe:
            df = DataLoader.load_instruments(
                data_dir, universe, self.start_time, self.end_time
            )
            self.instruments_set = set(zip(df["Instrument"], df["Date"]))

        self._setup_time_series()

    def _setup_time_series(self):
        self.data.index = self.data.index.swaplevel()
        self.data.sort_index(inplace=True)

        self._features = self.data[self.feature_names].to_numpy(copy=False)
        self._labels = (
            self.data[self.label_names].to_numpy(copy=False)
            if self.label_names
            else None
        )
        self._index = self.data.index

        slices = self._create_ts_slices(self._index, self.seq_len)

        daily_slices = defaultdict(list)
        for i, (code, date) in enumerate(self._index):
            # Skip if outside time window
            if date < self.start_time or date > self.end_time:
                continue

            # Skip if not in selected instruments
            if (
                self.instruments_set is not None
                and (code, date) not in self.instruments_set
            ):
                continue

            # Keep only slices with exact length
            slice = slices[i]
            if slice.stop - slice.start != self.seq_len:
                continue

            daily_slices[date].append(slice)

        self._daily_dates = list(daily_slices.keys())
        self._daily_slices = list(daily_slices.values())

        daily_counts = {date: len(slices) for date, slices in daily_slices.items()}
        print(f"Mode: {self.mode}. Sampled daily counts: {daily_counts}")

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
        features = np.stack([self._features[slice] for slice in slices])

        # Apply Robust Z-score normalization to selected feature columns
        features[:, :, self.stock_feature_indices] = ts_robust_zscore(
            features[:, :, self.stock_feature_indices], clip_outlier=True
        )

        # Fill missing features
        features = fillna(features, fill_value=0.0)

        data_dict = {
            "indices": indices,
            "industries": features[:, :, self.industry_indices],
            "stock_features": features[:, :, self.stock_feature_indices],
            "market_features": features[:, :, self.market_feature_indices],
        }

        if not self.label_names:
            return data_dict

        # Extract labels from the last time step of each sequence
        labels = np.array([self._labels[slice.stop - 1] for slice in slices])

        # In training mode, filter out samples with extreme label values
        if self.mode == "train":
            mask, labels = drop_extreme_label(labels)
            indices = indices[mask]
            features = features[mask]

        # Apply cross-sectional rank percentile normalization to labels
        ranks = labels.argsort(axis=0).argsort(axis=0)
        labels = ranks / (labels.shape[0] - 1)
        labels = labels.astype(np.float32)

        data_dict.update(
            {
                "indices": indices,
                "industries": features[:, :, self.industry_indices],
                "stock_features": features[:, :, self.stock_feature_indices],
                "market_features": features[:, :, self.market_feature_indices],
                "labels": labels,
            }
        )
        return data_dict

    def __len__(self):
        return len(self._daily_dates)
