import numpy as np

import pandas as pd
from torch.utils.data import Dataset

from aiq.utils.config import config as cfg

from .loader import DataLoader


class Dataset(Dataset):
    """
    Preparing data for model training and inference.
    """

    def __init__(
        self,
        data_dir,
        instruments,
        start_time=None,
        end_time=None,
        data_handler=None,
        training=False,
    ):
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

        # concat dataframes and set multi-index
        self.df = pd.concat(dfs, ignore_index=True)
        self.df = self.df.set_index(["Date", "Instrument"])

        # preprocess
        self.df = data_handler.process(self.df, training=training)

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
