import os
import pickle
from typing import List, Union

import numpy as np
import pandas as pd
import torch

from .loader import DataLoader


def load_instrument_data(data_dir, instruments, start_time, end_time, mode):
    is_train = mode == "train"
    instrument_name_to_idx = {}
    
    if not is_train:
        with open(os.path.join(data_dir, "instruments.pkl"), "rb") as f:
            instrument_name_to_idx = pickle.load(f)
    
    filtered_instruments = {}
    filtered_dfs = []
    next_id = 0
    
    for instrument in instruments:
        df = DataLoader.load_features(
            data_dir,
            instrument=instrument,
            start_time=start_time,
            end_time=end_time
        )

        if df is None or (not is_train and instrument not in instrument_name_to_idx):
            continue
        
        filtered_instruments[instrument] = next_id if is_train else instrument_name_to_idx[instrument]
        next_id += is_train  # 仅在训练模式下递增 ID
        filtered_dfs.append(df)
    
    if is_train:
        with open(os.path.join(data_dir, "instruments.pkl"), "wb") as f:
            pickle.dump(filtered_instruments, f)
    
    return filtered_instruments, filtered_dfs


class Dataset(torch.utils.data.Dataset):
    """
    Preparing data for model training and inference.
    """

    def __init__(
        self,
        data_dir,
        instruments,
        segments,
        data_handler=None,
        mode="train",
    ):
        # start and end time
        start_time, end_time = segments[mode]

        # load instruments of market
        if isinstance(instruments, str):
            instruments = DataLoader.load_instruments(
                data_dir, instruments, start_time, end_time
            )

        # load instrument's data
        instrument_ids, instrument_dfs = load_instrument_data(data_dir, instruments, start_time, end_time, mode)

        # extract feature and labels
        self.df = data_handler.process(instrument_dfs, mode=mode)

        self._feature_names = data_handler.feature_names
        self._label_names = data_handler.label_names

    def __getitem__(self, index):
        return self.df.iloc[[index]]

    def __len__(self):
        return self.df.shape[0]

    def insert(
        self,
        cols: List[str],
        data: Union[np.ndarray, List[np.ndarray], List[List[np.ndarray]]],
    ):
        self.df[cols] = data

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
        # sequence length and prediction length
        self.seq_len = seq_len
        self.mode = mode

        # start and end time
        start_time, end_time = segments[self.mode]

        # load instruments of market
        if isinstance(instruments, str):
            instruments = DataLoader.load_instruments(
                data_dir, instruments, start_time, end_time
            )

        # load instrument's data
        instrument_ids, instrument_dfs = load_instrument_data(data_dir, instruments, start_time, end_time, mode)

        # extract feature and labels
        self.df = data_handler.process(instrument_dfs, mode=mode)

        self._feature_names = data_handler.feature_names
        self._label_names = label_cols

        # change index to <code, date>
        self.df.index = self.df.index.swaplevel()
        self.df.sort_index(inplace=True)

        # data and index
        self._feature = self.df[self._feature_names].values.astype("float32")
        self._label = (
            self.df[self._label_names].values.astype("float32")
            if self._label_names is not None
            else None
        )
        self._index = self.df.index

        # create daily slices
        daily_slices = {
            date: [] for date in sorted(self._index.unique(level=1))
        }  # sorted by date
        self._batch_slices = self._create_ts_slices(self._index, self.seq_len)
        for i, (code, date) in enumerate(self._index):
            daily_slices[date].append((self._batch_slices[i], i, instrument_ids[code]))
        self._daily_slices = list(daily_slices.values())
        self._daily_index = list(
            daily_slices.keys()
        )  # index is the original date index

    def _create_ts_slices(self, index, seq_len):
        """
        create time series slices from pandas index

        Args:
            index (pd.MultiIndex): pandas multiindex with <instrument, datetime> order
            seq_len (int): sequence length
        """
        assert isinstance(index, pd.MultiIndex), "unsupported index type"
        assert seq_len > 0, "sequence length should be larger than 0"
        assert index.is_monotonic_increasing, "index should be sorted"

        # number of dates for each instrument
        sample_count_by_insts = index.to_series().groupby(level=0).size().values

        # start index for each instrument
        start_index_of_insts = np.roll(np.cumsum(sample_count_by_insts), 1)
        start_index_of_insts[0] = 0

        # all the [start, stop) indices of features
        # features between [start, stop) will be used to predict label at `stop - 1`
        slices = []
        for cur_loc, cur_cnt in zip(start_index_of_insts, sample_count_by_insts):
            for stop in range(1, cur_cnt + 1):
                end = cur_loc + stop
                start = max(end - seq_len, cur_loc)
                slices.append(slice(start, end))
        slices = np.array(slices, dtype="object")

        assert len(slices) == len(index)  # the i-th slice = index[i]

        return slices

    def padding_zeros(self, data, seq_len):
        """
        padding zeros to the end of data

        Args:
            data (np.ndarray): 2D array with shape (n, f) where n is sequence length and f is feature dimension
            seq_len (int): target sequence length
        """
        if data.shape[0] < seq_len:
            padding_zeros = np.zeros((seq_len - data.shape[0], data.shape[1]))
            return np.concatenate([padding_zeros, data], axis=0)
        else:
            return data

    def __getitem__(self, i):
        # index
        index = np.array([slice[1] for slice in self._daily_slices[i]])

        # instrument's id
        inst_ids = np.array([slice[2] for slice in self._daily_slices[i]])

        # feature
        feature = np.array(
            [
                self.padding_zeros(self._feature[slice[0]], self.seq_len)
                for slice in self._daily_slices[i]
            ]
        )
        feature = np.stack(feature)

        # label
        if self._label is not None:
            label = np.array(
                [self._label[slice[0].stop - 1] for slice in self._daily_slices[i]]
            )
            return index, inst_ids, feature, label
        else:
            return index, inst_ids, feature

    def __len__(self):
        return len(self._daily_index)

    @property
    def data(self):
        return self.df

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def label_names(self):
        return self._label_names


class MarketTSDataset(TSDataset):
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
        # sequence length and prediction length
        self.seq_len = seq_len
        self.mode = mode

        # start and end time
        start_time, end_time = segments[self.mode]

        # load instruments of market
        if isinstance(instruments, str):
            instruments = DataLoader.load_instruments(
                data_dir, instruments, start_time, end_time
            )

        # load instrument's data
        instrument_ids, instrument_dfs = load_instrument_data(data_dir, instruments, start_time, end_time, mode)

        # load market data
        market_dfs = {}
        for market_name in ["000300.SH", "000903.SH", "000905.SH"]:
            df = DataLoader.load_index_features(
                data_dir,
                instrument=market_name,
                start_time=start_time,
                end_time=end_time,
            )
            market_dfs[market_name] = df

        # extract feature and labels
        self.df = data_handler.process(
            instrument_dfs, market_dfs=market_dfs, mode=mode
        )

        self._feature_names = data_handler.feature_names
        self._label_names = label_cols

        # change index to <code, date>
        self.df.index = self.df.index.swaplevel()
        self.df.sort_index(inplace=True)

        # data and index
        self._feature = self.df[self._feature_names].values.astype("float32")
        self._label = (
            self.df[self._label_names].values.astype("float32")
            if self._label_names is not None
            else None
        )
        self._index = self.df.index

        # create daily slices
        daily_slices = {
            date: [] for date in sorted(self._index.unique(level=1))
        }  # sorted by date
        self._batch_slices = self._create_ts_slices(self._index, self.seq_len)
        for i, (code, date) in enumerate(self._index):
            daily_slices[date].append((self._batch_slices[i], i, instrument_ids[code]))
        self._daily_slices = list(daily_slices.values())
        self._daily_index = list(
            daily_slices.keys()
        )  # index is the original date index
