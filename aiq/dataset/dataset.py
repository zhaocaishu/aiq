import numpy as np
import pandas as pd
import torch

from .loader import DataLoader


def load_instruments_data(data_dir, instruments, start_time, end_time):
    if isinstance(instruments, str):
        instruments = DataLoader.load_instruments(
            data_dir, instruments, start_time, end_time
        )
    dfs = []
    for instrument in instruments:
        df = DataLoader.load_features(data_dir, instrument, start_time, end_time)
        if df is not None:
            dfs.append(df)
    return dfs


def load_market_data(data_dir, market_names, start_time, end_time):
    market_dfs = {}
    for market_name in market_names:
        df = DataLoader.load_index_features(data_dir, market_name, start_time, end_time)
        if df is not None:
            market_dfs[market_name] = df
    return market_dfs


class Dataset(torch.utils.data.Dataset):
    """
    Base Dataset for feature and label extraction.
    """

    def __init__(self, data_dir, instruments, segments, data_handler, mode="train"):
        self.data_dir = data_dir
        self.mode = mode
        self.data_handler = data_handler

        self.start_time, self.end_time = segments[self.mode]

        # Load instrument data
        self.instrument_dfs = load_instruments_data(
            self.data_dir, instruments, self.start_time, self.end_time
        )

    def prepare_data(self, instrument_dfs, market_dfs=None):
        self.df = self.data_handler.process(
            instrument_dfs, market_dfs=market_dfs, mode=self.mode
        )

        self._feature_names = self.data_handler.feature_names
        self._label_names = self.data_handler.label_names

        # Arrange index
        self.df.index = self.df.index.swaplevel()
        self.df.sort_index(inplace=True)

    def __getitem__(self, index):
        return self.df.iloc[[index]]

    def __len__(self):
        return len(self.df)

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
    Time Series dataset with fixed sequence length.
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
        self.label_cols = label_cols

        super().__init__(data_dir, instruments, segments, data_handler, mode)

        self.prepare_data(self.instrument_dfs)
        self._post_process()

    def _post_process(self):
        self._feature = self.df[self._feature_names].values.astype("float32")
        self._label = (
            self.df[self._label_names].values.astype("float32")
            if self._label_names is not None
            else None
        )
        self._index = self.df.index

        self._batch_slices = self._create_ts_slices(self._index, self.seq_len)

        daily_slices = {date: [] for date in sorted(self._index.unique(level=1))}
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

        return np.array(slices, dtype="object")

    def padding_zeros(self, data):
        if data.shape[0] < self.seq_len:
            padding = np.zeros((self.seq_len - data.shape[0], data.shape[1]))
            data = np.concatenate([padding, data], axis=0)
        return data

    def __getitem__(self, idx):
        index = np.array([s[1] for s in self._daily_slices[idx]])

        feature = np.stack(
            [
                self.padding_zeros(self._feature[slice[0]])
                for slice in self._daily_slices[idx]
            ]
        )

        if self._label is not None:
            label = np.array(
                [self._label[slice[0].stop - 1] for slice in self._daily_slices[idx]]
            )
            return index, feature, label
        else:
            return index, feature

    def __len__(self):
        return len(self._daily_index)


class MarketTSDataset(TSDataset):
    """
    Time Series Dataset with market features.
    """

    MARKET_NAMES = ["000300.SH", "000903.SH", "000905.SH"]

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
        super().__init__(
            data_dir, instruments, segments, seq_len, label_cols, data_handler, mode
        )

    def prepare_data(self, instrument_dfs, market_dfs=None):
        if market_dfs is None:
            market_dfs = load_market_data(
                self.data_dir, self.MARKET_NAMES, self.start_time, self.end_time
            )

        self.df = self.data_handler.process(
            instrument_dfs, market_dfs=market_dfs, mode=self.mode
        )

        self._feature_names = self.data_handler.feature_names
        self._label_names = self.label_cols

        self.df.index = self.df.index.swaplevel()
        self.df.sort_index(inplace=True)
