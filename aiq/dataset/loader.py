import os
from typing import List, Optional

import pandas as pd
import mysql.connector


class DataLoader:
    """Utility class for loading financial datasets from local directory or database."""

    @staticmethod
    def _read_csv(
        file_path: str, timestamp_col: str = "Date", start: str = "", end: str = "", columns: List[str] = None
    ) -> Optional[pd.DataFrame]:
        if not os.path.exists(file_path):
            return None

        df = pd.read_csv(file_path, usecols=columns)

        # 根据start和end日期过滤数据
        if start:
            df = df[df[timestamp_col] >= start]
        if end:
            df = df[df[timestamp_col] <= end]

        return df

    @staticmethod
    def _query_db(
        query: str, params: tuple, timestamp_col: str = "Date", columns: List[str] = None
    ) -> pd.DataFrame:
        conn = mysql.connector.connect(
            host="127.0.0.1",
            user="zcs",
            passwd="2025zcsdaydayup",
            database="stock_info",
        )
        try:
            df = pd.read_sql(query, conn, params=params, columns=columns)
            if timestamp_col in df.columns:
                df[timestamp_col] = pd.to_datetime(
                    df[timestamp_col].astype(str), format="%Y%m%d"
                ).dt.strftime("%Y-%m-%d")
            return df
        finally:
            conn.close()

    @staticmethod
    def load_instruments(
        data_dir: str, market_name: str, start_time: str = "", end_time: str = ""
    ) -> Optional[pd.DataFrame]:
        if data_dir:
            path = os.path.join(data_dir, "instruments", f"{market_name}.csv")
            df = DataLoader._read_csv(path, "Date", start_time, end_time)
        else:
            query = (
                "SELECT ts_code AS Instrument, trade_date AS Date "
                "FROM ts_idx_index_cons "
                "WHERE index_code=%s AND trade_date>=%s AND trade_date<=%s "
                "GROUP BY ts_code, trade_date"
            )
            df = DataLoader._query_db(
                query,
                (market_name, start_time.replace("-", ""), end_time.replace("-", "")),
                "Date"
            )
        return df

    @staticmethod
    def load_calendars(
        data_dir: str,
        timestamp_col: str = "Date",
        start_time: str = "",
        end_time: str = "",
    ) -> List[str]:
        if data_dir:
            path = os.path.join(data_dir, "calendars", "day.csv")
            df = DataLoader._read_csv(path, timestamp_col, start_time, end_time)
        else:
            query = (
                "SELECT DISTINCT exchange AS Exchange, cal_date AS Date "
                "FROM ts_basic_trade_cal "
                "WHERE is_open=1 AND cal_date >= %s AND cal_date <= %s"
            )
            df = DataLoader._query_db(
                query, (start_time.replace("-", ""), end_time.replace("-", "")), timestamp_col
            )

        return df[df["Exchange"] == "SSE"]["Date"].tolist() if df is not None else []

    @staticmethod
    def load_instrument_features(
        data_dir: str,
        instrument: str,
        timestamp_col: str = "Date",
        start_time: str = "",
        end_time: str = "",
        column_names: List[str] = None,
    ) -> Optional[pd.DataFrame]:
        if data_dir:
            path = os.path.join(data_dir, "features", f"{instrument}.csv")
            df = DataLoader._read_csv(path, timestamp_col, start_time, end_time, column_names)
        else:
            query = (
                "SELECT daily.ts_code AS Instrument, daily.trade_date AS Date, daily.close AS Close, "
                "daily.open AS Open, daily.high AS High, daily.low AS Low, daily.pre_close AS Pre_Close, "
                "daily.`change` AS `Change`, daily.pct_chg AS Pct_Chg, daily.volume AS Volume, daily.amount AS AMount, "
                "daily_basic.turnover_rate AS Turnover_rate, daily_basic.turnover_rate_f AS Turnover_rate_f, "
                "daily_basic.volume_ratio AS Volume_ratio, daily_basic.pe AS Pe, daily_basic.pe_ttm AS Pe_ttm, "
                "daily_basic.pb AS Pb, daily_basic.ps AS Ps, daily_basic.ps_ttm AS Ps_ttm, "
                "daily_basic.dv_ratio AS Dv_ratio, daily_basic.dv_ttm AS Dv_ttm, "
                "daily_basic.total_share AS Total_share, daily_basic.float_share AS Float_share, "
                "daily_basic.free_share AS Free_share, daily_basic.total_mv AS Total_mv, "
                "daily_basic.circ_mv AS Circ_mv, factor.adj_factor AS Adj_factor "
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
                timestamp_col,
                column_names,
            )

        if df is not None:
            df = df.sort_values(by=timestamp_col)
        return df

    @staticmethod
    def load_instruments_features(
        data_dir: str, instruments: List[str], start_time: str = "", end_time: str = ""
    ) -> pd.DataFrame:
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
        data_dir: str,
        market_name: str,
        timestamp_col: str = "Date",
        start_time: str = "",
        end_time: str = "",
        column_names: List[str] = None,
    ) -> Optional[pd.DataFrame]:
        if data_dir:
            path = os.path.join(data_dir, "features", f"{market_name}.csv")
            df = DataLoader._read_csv(path, timestamp_col, start_time, end_time, column_names)
        else:
            query = (
                "SELECT index_code AS Instrument, trade_date AS Date, close AS Close, "
                "open AS Open, high AS High, low AS Low, pre_close AS Pre_Close, "
                "`change` AS `Change`, pct_chg AS Pct_Chg, volume AS Volume, amount AS AMount "
                "FROM ts_idx_index_daily "
                "WHERE index_code = %s AND trade_date >= %s AND trade_date <= %s LIMIT 50000"
            )
            df = DataLoader._query_db(
                query,
                (market_name, start_time.replace("-", ""), end_time.replace("-", "")),
                timestamp_col,
                column_names,
            )

        if df is not None:
            df = df.sort_values(by=timestamp_col)
        return df

    @staticmethod
    def load_markets_features(
        data_dir: str, market_names: List[str], start_time: str = "", end_time: str = ""
    ) -> pd.DataFrame:
        dfs = [
            DataLoader.load_market_features(
                data_dir, market_name, start_time=start_time, end_time=end_time
            )
            for market_name in market_names
        ]
        dfs = [df for df in dfs if df is not None]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
