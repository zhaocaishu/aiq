import os
from typing import List, Optional

import pandas as pd
import mysql.connector


class DataLoader:
    """Utility class for loading financial datasets from local directory or database."""

    @staticmethod
    def _read_csv(
        file_path: str, timestamp_col: str, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        if not os.path.exists(file_path):
            return None

        df = pd.read_csv(file_path)

        # 根据start和end日期过滤数据
        if start:
            df = df[df[timestamp_col] >= start]
        if end:
            df = df[df[timestamp_col] <= end]

        return df

    @staticmethod
    def _query_db(
        query: str, params: tuple, columns: List[str], date_col: str = "Date"
    ) -> pd.DataFrame:
        conn = mysql.connector.connect(
            host="127.0.0.1",
            user="zcs",
            passwd="2025zcsdaydayup",
            database="stock_info",
        )
        try:
            df = pd.read_sql(query, conn, params=params)
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(
                    df[date_col].astype(str), format="%Y%m%d"
                ).dt.strftime("%Y-%m-%d")
            return df[columns]
        finally:
            conn.close()

    @staticmethod
    def load_instruments(data_dir, market, start_time=None, end_time=None) -> List[str]:
        if data_dir:
            path = os.path.join(data_dir, "instruments", f"{market}.csv")
            df = DataLoader._read_csv(path, "Date", start_time, end_time)
        else:
            query = (
                "SELECT ts_code AS Instrument, trade_date AS Date "
                "FROM ts_idx_index_weight "
                "WHERE index_code=%s AND trade_date>=%s AND trade_date<=%s "
                "GROUP BY ts_code, trade_date"
            )
            df = DataLoader._query_db(
                query,
                (market, start_time.replace("-", ""), end_time.replace("-", "")),
                ["Instrument", "Date"],
            )
        return df["Instrument"].unique().tolist() if df is not None else []

    @staticmethod
    def load_calendars(
        data_dir, timestamp_col="Trade_date", start_time=None, end_time=None
    ) -> Optional[List[str]]:
        if data_dir:
            path = os.path.join(data_dir, "calendars", "day.csv")
            df = DataLoader._read_csv(path, timestamp_col, start_time, end_time)
        else:
            query = (
                "SELECT DISTINCT exchange, DATE_FORMAT(cal_date, '%%Y-%%m-%%d') AS Trade_date "
                "FROM ts_basic_trade_cal "
                "WHERE is_open=1 AND cal_date >= %s AND cal_date <= %s"
            )
            df = DataLoader._query_db(
                query,
                (start_time.replace("-", ""), end_time.replace("-", "")),
                ["Exchange", "Trade_date"],
            )

        return (
            df[df["Exchange"] == "SSE"]["Trade_date"].tolist()
            if df is not None
            else None
        )

    @staticmethod
    def load_instrument_features(
        data_dir,
        instrument,
        timestamp_col="Date",
        start_time=None,
        end_time=None,
        column_names=None,
    ) -> Optional[pd.DataFrame]:
        if data_dir:
            path = os.path.join(data_dir, "features", f"{instrument}.csv")
            df = DataLoader._read_csv(path, timestamp_col, start_time, end_time)
        else:
            query = (
                "SELECT daily.*, daily_basic.turnover_rate, daily_basic.turnover_rate_f, "
                "daily_basic.volume_ratio, daily_basic.pe, daily_basic.pe_ttm, daily_basic.pb, "
                "daily_basic.ps, daily_basic.ps_ttm, daily_basic.dv_ratio, daily_basic.dv_ttm, "
                "daily_basic.total_share, daily_basic.float_share, daily_basic.free_share, "
                "daily_basic.total_mv, daily_basic.circ_mv, factor.adj_factor "
                "FROM ts_quotation_daily daily "
                "JOIN ts_quotation_daily_basic daily_basic ON "
                "daily.ts_code=daily_basic.ts_code AND daily.trade_date=daily_basic.trade_date "
                "JOIN ts_quotation_adj_factor factor ON "
                "daily.ts_code=factor.ts_code AND daily.trade_date=factor.trade_date "
                "WHERE daily.ts_code=%s AND daily.trade_date>=%s AND daily.trade_date<=%s LIMIT 50000"
            )
            df = DataLoader._query_db(
                query,
                (instrument, start_time.replace("-", ""), end_time.replace("-", "")),
                [
                    "Instrument",
                    "Date",
                    "Open",
                    "Close",
                    "High",
                    "Low",
                    "Pre_Close",
                    "Change",
                    "Pct_Chg",
                    "Volume",
                    "AMount",
                    "Turnover_rate",
                    "Turnover_rate_f",
                    "Volume_ratio",
                    "Pe",
                    "Pe_ttm",
                    "Pb",
                    "Ps",
                    "Ps_ttm",
                    "Dv_ratio",
                    "Dv_ttm",
                    "Total_share",
                    "Float_share",
                    "Free_share",
                    "Total_mv",
                    "Circ_mv",
                    "Adj_factor",
                ],
            )

        if df is not None:
            # 根据filter_list_date过滤上市前3个月的数据
            if "List_date" in df.columns:
                list_date = pd.to_datetime(df["List_date"].iloc[0])
                min_date = (list_date + pd.DateOffset(months=3)).strftime("%Y-%m-%d")
                df = df[df[timestamp_col] >= min_date]

            if column_names:
                df = df[column_names]

            df = df.sort_values(by=timestamp_col)
        return df

    @staticmethod
    def load_instruments_features(
        data_dir, instruments, start_time, end_time
    ) -> pd.DataFrame:
        if isinstance(instruments, str):
            instruments = DataLoader.load_instruments(
                data_dir, instruments, start_time, end_time
            )

        dfs = [
            DataLoader.load_instrument_features(
                data_dir, inst, start_time=start_time, end_time=end_time
            )
            for inst in instruments
        ]
        dfs = [df for df in dfs if df is not None]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    @staticmethod
    def load_market_features(
        data_dir,
        market,
        timestamp_col="Date",
        start_time=None,
        end_time=None,
        column_names=None,
    ) -> Optional[pd.DataFrame]:
        if data_dir:
            path = os.path.join(data_dir, "features", f"{market}.csv")
            df = DataLoader._read_csv(path, timestamp_col, start_time, end_time)
        else:
            query = (
                "SELECT * FROM ts_idx_index_daily "
                "WHERE index_code=%s AND trade_date>=%s AND trade_date<=%s LIMIT 50000"
            )
            df = DataLoader._query_db(
                query,
                (market, start_time.replace("-", ""), end_time.replace("-", "")),
                [
                    "Instrument",
                    "Date",
                    "Close",
                    "Open",
                    "High",
                    "Low",
                    "Pre_Close",
                    "Change",
                    "Pct_Chg",
                    "Volume",
                    "AMount",
                ],
            )

        if df is not None:
            if column_names:
                df = df[column_names]
            df = df.sort_values(by=timestamp_col)
        return df

    @staticmethod
    def load_markets_features(
        data_dir, markets: List[str], start_time: str, end_time: str
    ) -> pd.DataFrame:
        dfs = [
            DataLoader.load_market_features(
                data_dir, market, start_time=start_time, end_time=end_time
            )
            for market in markets
        ]
        dfs = [df for df in dfs if df is not None]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
