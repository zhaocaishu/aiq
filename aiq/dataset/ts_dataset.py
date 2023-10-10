import abc
import os
from typing import List

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from aiq.utils.date import date_add
from aiq.dataset.processor import TSStandardize
from aiq.ops import Ref

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

            # skip ticker of non-existed
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

            # prediction target
            df['Return'] = Ref(df['Close'], -5) / df['Close'] - 1
            df = df.dropna(subset=['Return'])

            dfs.append(df)

        # concat dataframes and set index
        self.df = pd.concat(dfs, ignore_index=True)

        # data pre-processing
        self.df.set_index('Symbol', inplace=True)
        ts_standardize = TSStandardize(target_cols=self.feature_names_, save_dir=self.save_dir)
        if self.training:
            ts_standardize.fit(self.df)
            self.df = ts_standardize(self.df)
        else:
            self.df = ts_standardize(self.df)
        self.df.reset_index(inplace=True)

        # build input and label data
        self.data = []
        trading_days = np.sort(self.df['Date'].unique())
        if self.label_names is not None:
            for i in range(seq_len, len(trading_days) - pred_len + 1):
                input_trade_days = trading_days[i - seq_len: i]
                pred_trade_days = trading_days[i: i + pred_len]
                self.data.append((input_trade_days, pred_trade_days))
        else:
            for i in range(seq_len, len(trading_days) + 1):
                input_trade_days = trading_days[i - seq_len: i]
                self.data.append(input_trade_days)

    @staticmethod
    def adjust_price(df):
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            df['Adj_' + col] = df[col] * df['Adj_factor']
        return df

    def to_dataframe(self):
        return self.df

    def __getitem__(self, index):
        if self.label_names is not None:
            input_trade_days, pred_trade_days = self.data[index]
            d_df = self.df[(self.df['Date'] >= input_trade_days[0]) & (self.df['Date'] <= pred_trade_days[-1])]

            symbol_day_count = d_df[['Symbol', 'Date']].groupby(['Symbol']).count()
            symbol_day_count.reset_index(inplace=True)

            inputs = []
            labels = []
            for idx, row in symbol_day_count.iterrows():
                symbol = row['Symbol']
                day_count = row['Date']
                if day_count != (self.seq_len + self.pred_len):
                    continue

                s_df = d_df[d_df['Symbol'] == symbol]
                input = torch.FloatTensor(s_df[self.feature_names].values[:self.seq_len, :])
                label = torch.FloatTensor(s_df[self.label_names].values[self.seq_len:, :])
                inputs.append(input)
                labels.append(label)

            inputs = torch.stack(inputs)
            labels = torch.stack(labels)
            return inputs, labels
        else:
            input_trade_days = self.data[index]
            d_df = self.df[(self.df['Date'] >= input_trade_days[0]) & (self.df['Date'] <= input_trade_days[-1])]

            symbol_day_count = d_df[['Symbol', 'Date']].groupby(['Symbol']).count()
            symbol_day_count.reset_index(inplace=True)

            inputs = []
            for idx, row in symbol_day_count.iterrows():
                symbol = row['Symbol']
                day_count = row['Date']
                if day_count != self.seq_len:
                    continue

                s_df = d_df[d_df['Symbol'] == symbol]
                input = torch.FloatTensor(s_df[self.feature_names].values[:self.seq_len, :])
                inputs.append(input)

            inputs = torch.stack(inputs)
            return inputs

    def __len__(self):
        return len(self.data)

    @property
    def feature_names(self):
        return self.feature_names_

    @property
    def label_names(self):
        return self.label_names_
