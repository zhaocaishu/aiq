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

        self.trading_calendar = DataLoader.load_calendar(
            self.data_dir,
            start_time=start_time,
            end_time=end_time,
        )

        self.logger = logger or logging.getLogger(__name__)

        # 初始化具体的策略驱动器
        self.backtester = TopKDropoutBacktester(
            top_k=self.top_k,
            date_col=self.date_col,
            instrument_col=self.instrument_col,
            up_limit_col=self.up_limit_col,
            down_limit_col=self.down_limit_col,
            score_col=self.pred_col,
            trading_calendar=self.trading_calendar,
            logger=self.logger,
            universe_col="IN_UNIVERSE",
        )

    def _compute_bench_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute benchmark 5D returns."""
        ret_5d = Ref(df["Close"], -5) / Ref(df["Close"], -1) - 1
        data = {
            self.date_col: df[self.date_col],
            self.instrument_col: df[self.instrument_col],
            self.label_col: ret_5d,
        }
        return pd.DataFrame(data)

    def _prepare_dataset(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        """Keep all quotes; a missing signal must not remove a holding's prices."""
        pred_df = pred_df[
            [self.date_col, self.instrument_col, self.pred_col, self.label_col]
        ].copy()
        pred_df[self.date_col] = pd.to_datetime(pred_df[self.date_col]).dt.strftime(
            "%Y-%m-%d"
        )

        key_cols = [self.date_col, self.instrument_col]
        universe = DataLoader.load_instruments(
            self.data_dir, self.benchmark, self.start_time, self.end_time
        )[key_cols].copy()
        universe[self.date_col] = pd.to_datetime(universe[self.date_col]).dt.strftime(
            "%Y-%m-%d"
        )

        # Keep full-period quotes for every potential holding, including dates
        # after it leaves the universe, so it can still be valued and sold.
        instruments = universe[self.instrument_col].unique().tolist()

        inst_features = DataLoader.load_instruments_features(
            self.data_dir, sorted(instruments), self.start_time, self.end_time
        )
        inst_features[self.date_col] = pd.to_datetime(
            inst_features[self.date_col]
        ).dt.strftime("%Y-%m-%d")
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

        # Load benchmark features.
        extended_end_time = (
            pd.to_datetime(self.end_time) + pd.Timedelta(days=20)
        ).strftime("%Y-%m-%d")
        bench_features = DataLoader.load_markets_features(
            self.data_dir, [self.benchmark], self.start_time, extended_end_time
        )

        bench_ret = self._compute_bench_returns(bench_features).rename(
            columns={self.label_col: "BENCH_RET_5D"}
        )[[self.date_col, "BENCH_RET_5D"]]
        bench_ret[self.date_col] = pd.to_datetime(bench_ret[self.date_col]).dt.strftime(
            "%Y-%m-%d"
        )

        # Quotes drive the simulation. Signals and benchmark returns are
        # optional observations, so neither is allowed to truncate the market
        # data. Leave missing signals as NaN; never forward-fill old scores.
        df = inst_features.merge(
            pred_df,
            on=key_cols,
            how="left",
            validate="one_to_one",
        ).merge(
            bench_ret,
            on=self.date_col,
            how="left",
            validate="many_to_one",
        )

        # Membership applies on the execution date; only prediction scores
        # are lagged. Out-of-universe quotes remain available to holdings.
        df["IN_UNIVERSE"] = pd.MultiIndex.from_frame(df[key_cols]).isin(
            pd.MultiIndex.from_frame(universe[key_cols])
        )

        return df

    def _compute_daily_ic(self, df: pd.DataFrame) -> pd.Series:
        df = df.loc[
            np.isfinite(df[self.label_col]),
            [self.date_col, self.pred_col, self.label_col],
        ]
        if df.empty:
            return pd.Series(dtype=float, name="RankIC")
        return df.groupby(self.date_col)[[self.pred_col, self.label_col]].apply(
            lambda x: x[self.pred_col].corr(x[self.label_col], method="spearman")
            if x[self.pred_col].nunique() > 1 and x[self.label_col].nunique() > 1
            else np.nan
        )

    def _compute_topk_stats(self, group: pd.DataFrame) -> dict:
        """Select before filtering labels; report statistics on observed labels."""
        k = self.top_k
        top_pred = group.nlargest(k, self.pred_col)
        observed = top_pred.loc[np.isfinite(top_pred[self.label_col])]
        reference = group.loc[np.isfinite(group[self.label_col])]
        stats = {
            f"HitRate@Top{k}": np.nan,
            f"WorstHit@Top{k}": np.nan,
            f"WinRate@Top{k}": np.nan,
            f"TopReturn@Top{k}": np.nan,
            f"ExcessReturn@Top{k}": np.nan,
            f"PoolExcess@Top{k}": np.nan,
            f"LabelCoverage@Top{k}": len(observed) / len(top_pred)
            if len(top_pred) > 0
            else np.nan,
        }
        if observed.empty:
            return stats

        top_return = observed[self.label_col].mean()
        stats[f"TopReturn@Top{k}"] = top_return
        # The pool baseline includes only the day's stocks with finite signals.
        stats[f"PoolExcess@Top{k}"] = top_return - reference[self.label_col].mean()

        benchmark_observed = observed.loc[np.isfinite(observed["BENCH_RET_5D"])]
        stats[f"ExcessReturn@Top{k}"] = (
            benchmark_observed[self.label_col] - benchmark_observed["BENCH_RET_5D"]
        ).mean()
        stats[f"WinRate@Top{k}"] = (
            benchmark_observed[self.label_col] > benchmark_observed["BENCH_RET_5D"]
        ).mean()

        # Undersized pools and constant labels cannot distinguish the two tails.
        if len(reference) >= 2 * k and reference[self.label_col].nunique() > 1:
            selected = set(observed[self.instrument_col])
            ranked = reference.sort_values(self.label_col, kind="stable")
            top_true = set(ranked.tail(k)[self.instrument_col])
            bottom_true = set(ranked.head(k)[self.instrument_col])
            stats[f"HitRate@Top{k}"] = len(selected & top_true) / len(observed)
            stats[f"WorstHit@Top{k}"] = len(selected & bottom_true) / len(observed)
        return stats

    def _compute_quantile_returns(self, group: pd.DataFrame) -> dict:
        """Assign score deciles before filtering labels; never split equal scores."""
        quantiles = np.ceil(
            group[self.pred_col].rank(method="average", pct=True) * 10
        )
        labels = group[self.label_col].where(np.isfinite(group[self.label_col]))
        returns = labels.groupby(quantiles).mean()
        stats = {f"Q{i}": returns.get(i, np.nan) for i in range(1, 11)}
        stats["Q10-Q1"] = stats["Q10"] - stats["Q1"]
        return stats

    def _compute_monthly_ic(self, daily_rank_ic: pd.Series) -> pd.DataFrame:
        daily_ic = pd.DataFrame({"RankIC": daily_rank_ic})
        months = pd.to_datetime(daily_ic.index).strftime("%Y-%m")
        monthly_ic = daily_ic.groupby(months).mean()
        monthly_ic["Days"] = daily_ic["RankIC"].groupby(months).count()
        monthly_ic.index.name = "Month"
        return monthly_ic.reset_index()

    def evaluate(self, pred_df: pd.DataFrame) -> str:
        """Evaluate scores against unnormalized, adjusted forward-return labels."""
        df = self._prepare_dataset(pred_df)

        # Only the current universe participates in signal diagnostics. Keep
        # all quotes, including missing signals, for portfolio valuation.
        # Each metric handles missing labels after determining its membership.
        metric_mask = df["IN_UNIVERSE"] & np.isfinite(df[self.pred_col])
        metric_df = df.loc[metric_mask].copy()

        self.logger.info(
            "统计评测：总样本 %d，候选样本 %d，有效标签 %d",
            len(df),
            len(metric_df),
            np.isfinite(metric_df[self.label_col]).sum(),
        )

        # 1. Daily RankIC, averaged with equal date weights.
        metric_dates = pd.Index(
            sorted(metric_df[self.date_col].unique()), name=self.date_col
        )
        daily_rank_ic = self._compute_daily_ic(metric_df).reindex(metric_dates)
        rank_ic = daily_rank_ic.mean()
        rank_ic_std = daily_rank_ic.std()
        rank_icir = rank_ic / rank_ic_std if rank_ic_std > 1e-12 else np.nan
        monthly_ic = self._compute_monthly_ic(daily_rank_ic)

        # 2. TopK selection and decile membership must not depend on whether a
        # future label is observable. Missing labels never trigger replacements.
        if metric_df.empty:
            topk_stats = self._compute_topk_stats(metric_df)
            daily_quantile_returns = pd.DataFrame(
                columns=[f"Q{i}" for i in range(1, 11)] + ["Q10-Q1"],
                dtype=float,
            )
        else:
            daily_topk_stats = pd.DataFrame(
                metric_df.groupby(self.date_col)
                .apply(self._compute_topk_stats)
                .tolist()
            )
            topk_stats = daily_topk_stats.mean().to_dict()
            daily_quantile_returns = pd.DataFrame(
                metric_df.groupby(self.date_col)
                .apply(self._compute_quantile_returns)
                .tolist()
            )

        quantile_returns = pd.DataFrame(
            {
                "MeanReturn": daily_quantile_returns.mean(),
                "Days": daily_quantile_returns.count(),
            }
        )
        quantile_returns.index.name = "Quantile"

        self.logger.info(
            "标签收益诊断：按日等权平均，未扣费、未年化；"
            "TopK 仅统计已选股票中标签可观测的部分，详见 LabelCoverage；"
            "HitRate/WorstHit 要求至少 2K 个标签可观测的候选股票。"
        )

        # 3. Portfolio Stats
        portfolio_stats = self.backtester.run(df)

        self.logger.info(f"{'═' * 72}")

        results = {
            "RankIC": rank_ic,
            "RankICIR": rank_icir,
            **topk_stats,
            **portfolio_stats,
        }

        summary = pd.DataFrame([results]).to_markdown(index=False, floatfmt=".4f")
        quantile_table = quantile_returns.reset_index().to_markdown(
            index=False, floatfmt=".4f"
        )
        monthly_table = monthly_ic.to_markdown(index=False, floatfmt=".4f")
        return (
            f"{summary}\n\n"
            "Forward-return diagnostics: before costs, unannualized; "
            "TopK statistics use observed labels (see LabelCoverage).\n\n"
            f"Quantile returns (Q1: lowest score, Q10: highest score):\n{quantile_table}\n\n"
            f"Monthly IC:\n{monthly_table}"
        )
