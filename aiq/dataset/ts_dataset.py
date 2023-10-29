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
        feature_names=None,
        label_names=None,
        adjust_price=True,
        cutoff_trade_days=90,
        min_trade_days=90,
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
        dfs = []
        for symbol, list_date in self.symbols:
            df = DataLoader.load_features(data_dir, symbol=symbol, start_time=start_time, end_time=end_time)

            # skip symbol of non-existed
            if df is None: continue

            # append ticker symbol
            df['Symbol'] = symbol

            # adjust price with factor
            if adjust_price:
                df = self.adjust_price(df)

            # keep data started from cutoff_trade_days after list date
            cur_start_time = date_add(list_date, n_days=cutoff_trade_days)
            if cur_start_time > start_time:
                df = df[(df['Date'] >= cur_start_time)]

            # check if symbol has enough trade days
            if df.shape[0] < min_trade_days: continue

            # features and target
            if adjust_price:
                close = df['Adj_Close']
            else:
                close = df['Close']

            df['ADV20'] = Mean(df['Volume'], 20)
            df['Return'] = close / Ref(close, 1) - 1
            df['Label'] = Ref(close, -5) / close - 1

            # process na
            df = df.dropna(subset=['Label'])
            df = df.fillna(0)

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

        symbols = self.df.index.unique().tolist()

        # build input and label data
        self.data = []
        for symbol in tqdm(symbols):
            s_df = self.df.loc[symbol]
            s_trading_days = np.sort(s_df['Date'].unique())
            s_df.set_index('Date', inplace=True)

            if self.label_names is not None:
                for i in range(seq_len, len(s_trading_days) - pred_len + 1):
                    input_label_trade_days = s_trading_days[i - seq_len: i + pred_len]
                    input_label_df = s_df.loc[input_label_trade_days]
                    input = torch.FloatTensor(input_label_df[self.feature_names_].values[:self.seq_len, :])
                    label = torch.FloatTensor(input_label_df[self.label_names_].values[self.seq_len:, :])
                    self.data.append((input, label))
            else:
                for i in range(seq_len, len(s_trading_days) + 1):
                    input_trade_days = s_trading_days[i - seq_len: i + pred_len]
                    input_df = s_df.loc[input_trade_days]
                    input = torch.FloatTensor(input_df[self.feature_names_].values[:self.seq_len, :])
                    self.data.append(input)

        # reset index
        self.df.reset_index(inplace=True)

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
        if self.label_names is not None:
            inputs, labels = self.data[index]
            return inputs, labels
        else:
            inputs = self.data[index]
            return inputs

    def __len__(self):
        return len(self.data)

    @property
    def feature_names(self):
        return self.feature_names_

    @property
    def label_names(self):
        return self.label_names_
