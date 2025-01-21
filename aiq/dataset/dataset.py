import numpy as np

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
