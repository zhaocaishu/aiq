import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from .loader import DataLoader


class Dataset(Dataset):
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

        # load instruments from market
        if isinstance(instruments, str):
            instruments = DataLoader.load_instruments(
                data_dir, instruments, start_time, end_time
            )

        # process instrument
        dfs = []
        for instrument in instruments:
            df = DataLoader.load_features(
                data_dir,
                instrument=instrument,
                start_time=start_time,
                end_time=end_time,
            )

            # skip instrument of non-existed
            if df is None:
                continue

            dfs.append(df)

        # preprocess
        self.df = data_handler.process(dfs, mode=mode)

        # feature and label names
        self.feature_names_ = data_handler.feature_names
        self.label_name_ = data_handler.label_name

    def to_dataframe(self):
        return self.df

    def add_column(self, name: str, data: np.array):
        self.df[name] = data

    @property
    def feature_names(self):
        return self.feature_names_

    @property
    def label_name(self):
        return self.label_name_

    def __getitem__(self, index):
        return self.df.iloc[[index]]

    def __len__(self):
        return self.df.shape[0]


def _create_ts_slices(index, seq_len):
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
            start = max(end - seq_len, 0)
            slices.append(slice(start, end))
    slices = np.array(slices, dtype="object")

    assert len(slices) == len(index)  # the i-th slice = index[i]

    return slices


class TSDataset(Dataset):
    """
    Time Series Dataset
    """

    def __init__(
        self,
        data_dir,
        instruments,
        segments,
        seq_len,
        data_handler=None,
        mode="train",
    ):
        # start and end time
        start_time, end_time = segments[mode]

        # load instruments from market
        if isinstance(instruments, str):
            instruments = DataLoader.load_instruments(
                data_dir, instruments, start_time, end_time
            )

        # process instrument
        dfs = []
        for instrument in instruments:
            df = DataLoader.load_features(
                data_dir,
                instrument=instrument,
                start_time=start_time,
                end_time=end_time,
            )

            # skip instrument of non-existed
            if df is None:
                continue

            dfs.append(df)

        # preprocess
        self.df = data_handler.process(dfs, mode=mode)

        # pre-fetch data and change index to <code, date>
        self.df.index = self.df.index.swaplevel()
        self.df.sort_index(inplace=True)

        # create batch slices
        self.seq_len = seq_len
        self._index = self.df.index
        self._batch_slices = _create_ts_slices(self._index, self.seq_len)

        # create daily slices
        daily_slices = {
            date: [] for date in sorted(self._index.unique(level=1))
        }  # sorted by date
        for i, (code, date) in enumerate(self._index):
            daily_slices[date].append(self._batch_slices[i])
        self._daily_slices = np.array(list(daily_slices.values()), dtype="object")
        self._daily_index = pd.Series(
            list(daily_slices.keys())
        )  # index is the original date index

        # feature and label names
        self.feature_names_ = data_handler.feature_names
        self.label_name_ = data_handler.label_name

    @property
    def feature_names(self):
        return self.feature_names_

    @property
    def label_name(self):
        return self.label_name_

    def __getitem__(self, index):
        return self._daily_slices[index]

    def __len__(self):
        return len(self._daily_index)
