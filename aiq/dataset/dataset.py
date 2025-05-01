import numpy as np
import pandas as pd
import torch

from aiq.utils.processing import drop_extreme_label, zscore

from .loader import DataLoader


class Dataset(torch.utils.data.Dataset):
    """
    Preparing data for model training and inference.
    """

    def __init__(
        self, data_dir, instruments, segments, data_handler=None, mode="train"
    ):
        start_time, end_time = segments[mode]
        instrument_dfs = DataLoader.load_instruments_features(
            data_dir, instruments, start_time, end_time
        )
        self.df = data_handler.process(instrument_dfs, mode=mode)
        self._feature_names = data_handler.feature_names
        self._label_names = data_handler.label_names

    def __getitem__(self, index):
        return self.df.iloc[[index]]

    def __len__(self):
        return self.df.shape[0]

    @property
    def data(self):
        return self.df

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
        data_dir,
        instruments,
        segments,
        seq_len,
        label_cols=None,
        data_handler=None,
        mode="train",
    ):
        self.seq_len = seq_len
        self.mode = mode
        start_time, end_time = segments[self.mode]
        instrument_df = DataLoader.load_instruments_features(
            data_dir, instruments, start_time, end_time
        )
        self.df = data_handler.process(instrument_df, mode=mode)
        self.data_handler = data_handler
        self.label_cols = label_cols
        self._setup_time_series()

    def _setup_time_series(self):
        self._feature_names = self.data_handler.feature_names
        self._label_names = self.label_cols
        self.df.index = self.df.index.swaplevel()
        self.df.sort_index(inplace=True)
        self._feature = self.df[self._feature_names].values.astype("float32")
        self._label = (
            self.df[self._label_names].values.astype("float32")
            if self._label_names is not None
            else None
        )
        self._index = self.df.index
        daily_slices = {date: [] for date in sorted(self._index.unique(level=1))}
        self._batch_slices = self._create_ts_slices(self._index, self.seq_len)
        for i, (code, date) in enumerate(self._index):
            daily_slices[date].append((self._batch_slices[i], i))
        self._daily_slices = list(daily_slices.values())
        self._daily_index = list(daily_slices.keys())

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
            valid_mask, filtered_labels = drop_extreme_label(labels)
            features = features[valid_mask]
            labels = filtered_labels

        normalized_labels = zscore(labels).reshape(-1, 1)

        return sample_indices, features, normalized_labels

    def __len__(self):
        return len(self._daily_index)


class MarketTSDataset(TSDataset):
    """
    Time series dataset with market data.
    """

    def __init__(
        self,
        data_dir,
        instruments,
        segments,
        seq_len,
        label_cols=None,
        data_handler=None,
        mode="train",
    ):
        self.seq_len = seq_len
        self.mode = mode
        start_time, end_time = segments[self.mode]
        instrument_df = DataLoader.load_instruments_features(
            data_dir, instruments, start_time, end_time
        )
        market_names = ["000300.SH", "000903.SH", "000905.SH"]
        market_df = DataLoader.load_markets_features(
            data_dir, market_names, start_time, end_time
        )
        self.df = data_handler.process(instrument_df, market_df=market_df, mode=mode)
        self.data_handler = data_handler
        self.label_cols = label_cols
        self._setup_time_series()
