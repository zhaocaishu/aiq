import abc
import os
import time
from typing import List
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from aiq.utils.date import date_add
from aiq.dataset.processor import TSStandardize
from aiq.ops import Ref, Mean

from .loader import DataLoader


class TSDataset(Dataset):
    """
    Preparing time series data for model training and inference.
    """

    def __init__(
        self,
        data_dir,
        instruments,
        save_dir,
        start_time=None,
        end_time=None,
        segment=None,
        feature_names=None,
        label_names=None,
        adjust_price=True,
        min_trade_days=63,
        seq_len=60,
        pred_len=6,
        training=True
    ):
        # feature and label column names
        self.feature_names_ = feature_names
        self.label_names_ = label_names

        # input and prediction sequence length
        self.seq_len = seq_len
        self.pred_len = pred_len

        # options
        self.training = training
        self.save_dir = save_dir

        # symbol's name and list date
        self.symbols = DataLoader.load_symbols(data_dir, instruments, start_time=start_time, end_time=end_time)

        # process per symbol
        symbols = []
        dfs = []
        for symbol, list_date in self.symbols:
            df = DataLoader.load_features(data_dir, symbol=symbol, start_time=start_time, end_time=end_time,
                                          column_names=['Symbol', 'Date', 'Open', 'Close', 'High', 'Low', 'Volume',
                                                        'Adj_factor'])

            # skip symbol of non-existed
            if df is None: continue

            # adjust price with factor
            if adjust_price:
                df = self.adjust_price(df)

            # keep data started from min_trade_days after list date
            cur_start_time = date_add(list_date, n_days=min_trade_days)
            if cur_start_time > start_time:
                df = df[(df['Date'] >= cur_start_time)]

            # check if symbol has enough trade days
            if df.shape[0] < min_trade_days: continue

            # features
            if adjust_price:
                close = df['Adj_Close']
            else:
                close = df['Close']

            df['ADV20'] = Mean(df['Volume'], 20)
            df['Return'] = close / Ref(close, 1) - 1
            df[self.feature_names_] = df[self.feature_names_].fillna(0)

            # target
            if self.label_names_ is not None:
                df['Label'] = Ref(close, -5) / close - 1
                df = df.dropna(subset=['Label'])

            symbols.append(symbol)
            dfs.append(df)

        # concat dataframes and set index
        self.df = pd.concat(dfs, ignore_index=True)
        self.df.set_index('Symbol', inplace=True)

        # data pre-processing
        ts_standardize = TSStandardize(target_cols=self.feature_names_, save_dir=self.save_dir)
        if self.training:
            ts_standardize.fit(self.df)
            self.df = ts_standardize(self.df)
        else:
            self.df = ts_standardize(self.df)

        # build input and label data
        if self.label_names_ is not None:
            data = {'Symbol': [], 'Date': [], 'Close': [], 'Feature': [], 'Label': []}
        else:
            data = {'Symbol': [], 'Date': [], 'Close': [], 'Feature': []}

        for symbol in tqdm(symbols):
            s_df = self.df.loc[symbol]
            s_trade_dates = np.sort(s_df['Date'].unique())
            s_df.set_index('Date', inplace=True)

            if self.label_names_ is not None:
                for i in range(seq_len, len(s_trade_dates) - pred_len + 1):
                    trade_date = s_trade_dates[i]
                    if trade_date < segment[0] or trade_date > segment[1]:
                        continue

                    feature_label_trade_dates = s_trade_dates[i - seq_len: i + pred_len]
                    feature_label_df = s_df.loc[feature_label_trade_dates]
                    feature = feature_label_df[self.feature_names_].values[:self.seq_len, :].astype(np.float32)
                    label = feature_label_df[self.label_names_].values[self.seq_len:, :].astype(np.float32)
                    data['Symbol'].append(symbol)
                    data['Date'].append(trade_date)
                    data['Close'].append(s_df.loc[trade_date, 'Close'])
                    data['Feature'].append(feature)
                    data['Label'].append(label)
            else:
                for i in range(seq_len, len(s_trade_dates)):
                    trade_date = s_trade_dates[i]
                    if trade_date < segment[0] or trade_date > segment[1]:
                        continue

                    feature_trade_dates = s_trade_dates[i - seq_len: i]
                    feature_df = s_df.loc[feature_trade_dates]
                    feature = feature_df[self.feature_names_].values[:self.seq_len, :].astype(np.float32)
                    data['Symbol'].append(symbol)
                    data['Date'].append(trade_date)
                    data['Close'].append(s_df.loc[trade_date, 'Close'])
                    data['Feature'].append(feature)

        # reset index
        self.df = pd.DataFrame(data)

    @staticmethod
    def adjust_price(df):
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            df['Adj_' + col] = df[col] * df['Adj_factor']
        return df

    def add_column(self, name: str, data: np.array):
        self.df[name] = data

    def to_dataframe(self):
        return self.df

    def __getitem__(self, index):
        if self.label_names_ is not None:
            feature = torch.FloatTensor(self.df.iloc[index]['Feature'])
            label = torch.FloatTensor(self.df.iloc[index]['Label'])
            return feature, label
        else:
            feature = torch.FloatTensor(self.df.iloc[index]['Feature'])
            return feature

    def __len__(self):
        return self.df.shape[0]

    @property
    def feature_names(self):
        return self.feature_names_

    @property
    def label_names(self):
        return self.label_names_
