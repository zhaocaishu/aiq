import logging
import numpy as np
import pandas as pd

from aiq.dataset.loader import DataLoader
from aiq.ops import Ref

from .backtester import TopKDropoutBacktester


class Evaluator:
    """
    Offline evaluation framework for A-share multi-asset prediction models.
    Delegates backtesting execution to specialized strategy instances inside `backtester`.
    """

    def __init__(
        self,
        data_dir: str,
        start_time: str,
        end_time: str,
        benchmark: str = "000905.SH",
        date_col: str = "Date",
        instrument_col: str = "Instrument",
        up_limit_col: str = "Up_limit",
        down_limit_col: str = "Down_limit",
        pred_col: str = "PRED_RET_5D",
        label_col: str = "RET_5D",
        top_k: int = 30,
        logger: logging.Logger = None,
    ):
        self.data_dir = data_dir
        self.start_time = start_time
        self.end_time = end_time
        self.benchmark = benchmark

        self.date_col = date_col
        self.instrument_col = instrument_col
        self.up_limit_col = up_limit_col
        self.down_limit_col = down_limit_col
        self.pred_col = pred_col
        self.label_col = label_col
        self.top_k = top_k

        self.logger = logger or logging.getLogger(__name__)

        # 初始化具体的策略驱动器
        self.backtester = TopKDropoutBacktester(
            top_k=self.top_k,
            date_col=self.date_col,
            instrument_col=self.instrument_col,
            up_limit_col=self.up_limit_col,
            down_limit_col=self.down_limit_col,
            pred_col=self.pred_col,
            label_col=self.label_col,
            logger=self.logger,
        )

    def _compute_bench_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute benchmark 5D returns."""
        df["VWAP"] = df["Amount"] / (df["Volume"] + 1e-12) * 10
        ret_5d = Ref(df["VWAP"], -5) / Ref(df["VWAP"], -1) - 1
        data = {
            self.date_col: df[self.date_col],
            self.instrument_col: df[self.instrument_col],
            self.label_col: ret_5d,
        }
        return pd.DataFrame(data)

    def _prepare_dataset(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        """Align prediction / label / benchmark returns."""
        instruments = (
            DataLoader.load_instruments(
                self.data_dir, self.benchmark, self.start_time, self.end_time
            )[self.instrument_col]
            .unique()
            .tolist()
        )

        inst_features = DataLoader.load_instruments_features(
            self.data_dir, instruments, self.start_time, self.end_time
        )
        inst_features["VWAP"] = (
            inst_features["AMount"] / (inst_features["Volume"] + 1e-12) * 10
        )
        inst_features = inst_features[
            [
                self.date_col,
                self.instrument_col,
                self.up_limit_col,
                self.down_limit_col,
                "Close",
                "High",
                "Low",
                "VWAP",
            ]
        ]

        extended_end_time = (
            pd.to_datetime(self.end_time) + pd.Timedelta(days=20)
        ).strftime("%Y-%m-%d")
        bench_features = DataLoader.load_markets_features(
            self.data_dir, [self.benchmark], self.start_time, extended_end_time
        )

        bench_ret = self._compute_bench_returns(bench_features).rename(
            columns={self.label_col: "BENCH_RET_5D"}
        )[[self.date_col, "BENCH_RET_5D"]]

        df = inst_features.merge(
            pred_df[
                [self.date_col, self.instrument_col, self.pred_col, self.label_col]
            ],
            on=[self.date_col, self.instrument_col],
            how="inner",
        ).merge(bench_ret, on=self.date_col, how="inner")

        assert set(df[self.date_col]) == set(
            pred_df[self.date_col]
        ), f"{self.date_col} mismatch"
        assert set(df[self.instrument_col]) == set(
            pred_df[self.instrument_col]
        ), f"{self.instrument_col} mismatch"
        assert not pred_df[self.pred_col].isna().any(), f"{self.pred_col} contains NaN"

        return df

    def _compute_daily_ic(self, df: pd.DataFrame) -> pd.Series:
        return df.groupby(self.date_col).apply(
            lambda x: x[self.pred_col].corr(x[self.label_col], method="spearman")
        )

    def _compute_hit_rate(self, group: pd.DataFrame) -> dict:
        k = self.top_k
        top_pred = set(group.nlargest(k, self.pred_col)[self.instrument_col])
        top_true = set(group.nlargest(k, self.label_col)[self.instrument_col])
        return {f"HitRate@Top{k}": len(top_pred & top_true) / k}

    def _compute_win_rate(self, group: pd.DataFrame) -> dict:
        k = self.top_k
        top_pred = group.nlargest(k, self.pred_col)
        win_rate = (top_pred[self.label_col] > top_pred["BENCH_RET_5D"]).mean()
        return {f"WinRate@Top{k}": win_rate}

    def evaluate(self, pred_df: pd.DataFrame) -> str:
        """Run statistical cross-sectional evaluation and simulation."""
        df = self._prepare_dataset(pred_df)

        # 1. IC & ICIR
        daily_ic = self._compute_daily_ic(df).dropna()
        ic = daily_ic.mean()
        icir = ic / daily_ic.std() if daily_ic.std() > 1e-12 else np.nan

        # 2. Hit Rate
        daily_hr = pd.DataFrame(
            df.groupby(self.date_col).apply(self._compute_hit_rate).tolist()
        )
        hr_stats = daily_hr.mean().to_dict()

        # 3. Win Rate
        daily_wr = pd.DataFrame(
            df.groupby(self.date_col).apply(self._compute_win_rate).tolist()
        )
        wr_stats = daily_wr.mean().to_dict()

        # 4. 执行回测
        portfolio_stats = self.backtester.run(df)

        self.logger.info(f"{'═' * 72}")

        results = {
            "IC": ic,
            "ICIR": icir,
            **hr_stats,
            **wr_stats,
            **portfolio_stats,
        }

        return pd.DataFrame([results]).to_markdown(index=False, floatfmt=".4f")
