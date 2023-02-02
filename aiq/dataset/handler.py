import abc

import pandas as pd
import talib as ta


class DataHandler(abc.ABC):
    def __init__(self):
        pass

    def fetch(self, df: pd.DataFrame = None) -> pd.DataFrame:
        pass

class Alpha100(DataHandler):
    def __init__(self, test_mode=False):
        self.test_mode = test_mode

    def fetch(self, df: pd.DataFrame = None) -> pd.DataFrame:
        close = df['Close']
        volume = df['Volume']

        momentum_1m = ta.MOM(close, timeperiod=30)
        momentum_3m = ta.MOM(close, timeperiod=90)
        momentum_6m = ta.MOM(close, timeperiod=180)
        momentum_12m = ta.MOM(close, timeperiod=360)
        momentum_24m = ta.MOM(close, timeperiod=720)
        df['momentum_1m'] = momentum_1m
        df['momentum_3m'] = momentum_3m
        df['momentum_6m'] = momentum_6m
        df['momentum_12m'] = momentum_12m
        df['momentum_24m'] = momentum_24m

        highlow_1m = close.rolling(window=30).max() / close.rolling(window=30).min()
        highlow_3m = close.rolling(window=90).max() / close.rolling(window=90).min()
        highlow_6m = close.rolling(window=180).max() / close.rolling(window=180).min()
        df['highlow_1m'] = highlow_1m
        df['highlow_3m'] = highlow_3m
        df['highlow_6m'] = highlow_6m

        vstd_1m = volume.rolling(window=30).std()
        vstd_3m = volume.rolling(window=90).std()
        vstd_6m = volume.rolling(window=180).std()
        df['vstd_1m'] = vstd_1m
        df['vstd_3m'] = vstd_3m
        df['vstd_6m'] = vstd_6m

        sobv = ta.OBV(close, volume)
        rsi = ta.RSI(close, timeperiod=14)
        macd, _, _ = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['sobv'] = sobv
        df['rsi'] = rsi
        df['macd'] = macd

        if not self.test_mode:
            df['label'] = close.diff(periods=1)
            df = df.dropna(subset=['label'])

        return df
