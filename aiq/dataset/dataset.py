import numpy as np
import pandas as pd
import torch

from aiq.utils.processing import drop_extreme_label, zscore


class Dataset(torch.utils.data.Dataset):
    """
    Preparing data for model training and inference.
    """

    def __init__(self, segments, data, feature_names, label_names, mode="train"):
        start_time, end_time = segments[mode]
        self._data = data[(data["Date"] >= start_time) & (data["Date"] <= end_time)]
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
        data,
        segments,
        seq_len,
        feature_names=None,
        label_names=None,
        mode="train",
    ):
        self.seq_len = seq_len
        self.mode = mode
        self.start_time, self.end_time = segments[self.mode]
        self._feature_names = feature_names
        self._label_names = label_names
        self._data = data.copy()
        self._setup_time_series()

    def _setup_time_series(self):
        self._data.index = self._data.index.swaplevel()
        self._data.sort_index(inplace=True)
        self._feature = self._data[self._feature_names].values.astype("float32")
        self._label = (
            self._data[self._label_names].values.astype("float32")
            if self._label_names is not None
            else None
        )
        self._index = self._data.index
        daily_slices = {date: [] for date in sorted(self._index.unique(level=1))}
        self._batch_slices = self._create_ts_slices(self._index, self.seq_len)
        for i, (code, date) in enumerate(self._index):
            daily_slices[date].append((self._batch_slices[i], i))
        self._daily_slices = np.array(list(daily_slices.values()), dtype="object")
        self._daily_index = pd.Series(list(daily_slices.keys()))

        mask = (self._daily_index.values >= self.start_time) & (self._daily_index.values <= self.end_time)
        self._daily_slices = self._daily_slices[mask]
        self._daily_index = self._daily_index[mask]

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

    def padding_zeros(self, data, seq_len):
        if data.shape[0] < seq_len:
            padding_zeros = np.zeros((seq_len - data.shape[0], data.shape[1]))
            return np.concatenate([padding_zeros, data], axis=0)
        return data

    def __getitem__(self, index):
        """根据索引获返回样本索引、特征和标准化后的标签（若存在）"""
        daily_slices = self._daily_slices[index]

        sample_indices = np.array([slice[1] for slice in daily_slices])

        features = [
            self.padding_zeros(self._feature[slice[0]], self.seq_len)
            for slice in daily_slices
        ]
        features = np.stack(features)

        if self._label is None:
            return sample_indices, features

        labels = np.array(
            [self._label[slice[0].stop - 1] for slice in daily_slices]
        ).squeeze()

        if self.mode == "train":
            mask, labels = drop_extreme_label(labels)
            features = features[mask]

        normalized_labels = zscore(labels).reshape(-1, 1)

        return sample_indices, features, normalized_labels

    def __len__(self):
        return len(self._daily_index)
