import abc
import os
from typing import List

import numpy as np
import pandas as pd

from aiq.ops import Rank, Cov, Mean, Std, Corr, CSRank, Sum, Max, Min

from .loader import DataLoader
from .processor import CSFillna, CSNeutralize, CSFilter, CSZScore


class Dataset(abc.ABC):
    """
    Preparing data for model training and inference.
    """

    def __init__(
        self,
        data_dir,
        instruments,
        start_time=None,
        end_time=None,
        handler=None,
        adjust_price=True,
        training=False
    ):
        # feature and label names
        self.feature_names_ = None
        self.label_name_ = None

        # symbol of instruments
        with open(os.path.join(data_dir, 'instruments/%s.txt' % instruments), 'r') as f:
            self.symbols = [line.strip().split()[0] for line in f.readlines()]

        # process per symbol
        dfs = []
        symbols = []
        for symbol in self.symbols:
            df = DataLoader.load(os.path.join(data_dir, 'features'), symbol=symbol, start_time=start_time,
                                 end_time=end_time)

            # skip ticker of non-existed or small periods
            if df is None:
                continue

            # append ticker symbol
            df['Symbol'] = symbol
            symbols.append(symbol)

            # adjust price with factor
            if adjust_price:
                df = self.adjust_price(df)

            # extract ticker factors
            if handler is not None:
                df = handler.fetch(df)

            dfs.append(df)

        # concat dataframes and set index
        self.df = pd.concat(dfs, ignore_index=True)
        self.df = self.df.set_index(['Date', 'Symbol'])

        self.df['VWAP_CSRANK'] = CSRank((self.df['High'] + self.df['Low']) / 2.0)
        self.df['CLOSE_CSRANK'] = CSRank(self.df['Close'])
        self.df['VOLUME_CSRANK'] = CSRank(self.df['Volume'])
        self.df['OPEN_CSRANK'] = CSRank(self.df['Open'])
        self.df['HIGH_CSRANK'] = CSRank(self.df['High'])

        def ts_func_lv1(x):
            x['OVRANKCORR10'] = -1.0 * Corr(x['OPEN_CSRANK'], x['VOLUME_CSRANK'], 10)
            x['CVRANKCOV5'] = Cov(x['CLOSE_CSRANK'], x['VOLUME_CSRANK'], 5)
            x['HVRANKCORR3'] = Corr(x['HIGH_CSRANK'], x['VOLUME_CSRANK'], 5)
            x['HVRANKCORR5'] = -1.0 * Corr(x['High'], x['VOLUME_CSRANK'], 3)
            x['HVRANKCOV5'] = Cov(x['HIGH_CSRANK'], x['VOLUME_CSRANK'], 5)
            x['WVRANKCORR5'] = Corr(x['HIGH_CSRANK'], x['VOLUME_CSRANK'], 5)
            return x
        self.df = self.df.groupby('Symbol', group_keys=False).apply(ts_func_lv1)
        self.df['CVRANKCOV5'] = -1.0 * CSRank(self.df['CVRANKCOV5'])
        self.df['HVRANKCORR3'] = CSRank(self.df['HVRANKCORR3'])
        self.df['HVRANKCOV5'] = -1.0 * CSRank(self.df['HVRANKCOV5'])
        self.df['WVRANKCORR5'] = CSRank(self.df['WVRANKCORR5'])

        def ts_func_lv2(x):
            x['HVRANKCORR3'] = -1.0 * Sum(x['HVRANKCORR3'], 3)
            x['WVRANKCORR5'] = -1.0 * Max(x['WVRANKCORR5'], 5)
            x['CHLRANKCORR12'] = (x['Close'] - Min(x['Low'], 12)) / (Max(x['High'], 12) - Min(x['Low'], 12))
            return x
        self.df = self.df.groupby('Symbol', group_keys=False).apply(ts_func_lv2)
        self.df['CHLRANKCORR12'] = CSRank(self.df['CHLRANKCORR12'])

        def ts_func_lv3(x):
            x['CHLRANKCORR12'] = -1.0 * Corr(x['CHLRANKCORR12'], x['VOLUME_CSRANK'], 6)
            return x
        self.df = self.df.groupby('Symbol', group_keys=False).apply(ts_func_lv3)

        self.feature_names_ = ['OVRANKCORR10', 'CVRANKCOV5', 'HVRANKCORR3', 'HVRANKCORR5', 'HVRANKCOV5',
                               'WVRANKCORR5', 'CHLRANKCORR12']

        # assign features and label name
        if handler is not None:
            self.feature_names_ += handler.feature_names
            self.label_name_ = handler.label_name

        # preprocessors
        if self.feature_names_ is not None and False:
            # fill nan
            fillna = CSFillna(target_cols=self.feature_names_)
            self.df = fillna(self.df)

            # remove outlier
            outlier_filter = CSFilter(target_cols=self.feature_names_)
            self.df = outlier_filter(self.df)

            # factor neutralize
            cs_neut = CSNeutralize(industry_num=110, industry_col='Industry_id', market_cap_col='Total_mv',
                                   target_cols=self.feature_names_)
            self.df = cs_neut(self.df)

            # factor standardization
            cs_score = CSZScore(target_cols=self.feature_names_)
            self.df = cs_score(self.df)

        # reset index
        self.df.reset_index(inplace=True)

        # random shuffle
        if training:
            self.df = self.df.sample(frac=1.0)

    @staticmethod
    def adjust_price(df):
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            df[col] = df[col] * df['Adj_factor']
        return df

    def to_dataframe(self):
        return self.df

    def add_column(self, name: str, data: np.array):
        self.df[name] = data

    def slice(self, start_time, end_time):
        return self.df[(self.df['Date'] >= start_time) & (self.df['Date'] <= end_time)].copy()

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


class Subset(Dataset):
    def __init__(self, dataset, start_time, end_time):
        self.feature_names_ = dataset.feature_names_
        self.label_name_ = dataset.label_name_
        self.df = dataset.slice(start_time, end_time)


def ts_split(dataset: Dataset, segments: List[List[str]]):
    return [Subset(dataset, segment[0], segment[1]) for segment in segments]
