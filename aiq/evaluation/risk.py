import numpy as np
import pandas as pd


class RiskEvaluator:
    """Ex-post signal diagnostics, not attribution of actual portfolio P&L.

    Inputs contain raw forward returns and point-in-time risk characteristics.
    Every statistic is calculated within a date before averaging across dates.
    No significance statistics are reported because forward labels overlap.
    """

    FACTORS = ("SIZE", "VOL20", "TURN20", "BETA60")
    CONTRIBUTIONS = ("Industry", "SIZE", "VOL20", "TURN20", "BETA60", "Residual")

    def __init__(
        self,
        top_k=30,
        date_col="Date",
        instrument_col="Instrument",
        pred_col="PRED_RET_5D",
        label_col="RET_5D",
        min_group_size=30,
        min_regression_size=100,
    ):
        self.top_k = top_k
        self.date_col = date_col
        self.instrument_col = instrument_col
        self.pred_col = pred_col
        self.label_col = label_col
        self.min_group_size = min_group_size
        self.min_regression_size = min_regression_size

    def prepare_features(self, quotes, benchmark, calendar):
        """Build trailing risks on a common calendar, without filling missing quotes.

        Signals are assumed to be formed after the close. A 60-return window
        requires 61 observed adjusted closes. Missing sessions remain missing,
        so pct_change never silently turns a multi-session gap into a daily return.
        """
        keys = [self.date_col, self.instrument_col]
        quotes = quotes.copy()
        benchmark = benchmark.copy()
        for frame in (quotes, benchmark):
            frame[self.date_col] = pd.to_datetime(frame[self.date_col]).dt.normalize()
        if quotes.duplicated(keys).any() or benchmark[self.date_col].duplicated().any():
            raise ValueError(
                "Risk inputs must have unique stock/date and benchmark/date keys"
            )
        dates = (
            pd.DatetimeIndex(pd.to_datetime(calendar))
            .normalize()
            .unique()
            .sort_values()
        )
        market_close = benchmark.set_index(self.date_col)["Close"].reindex(dates)
        market_close = market_close.where(
            np.isfinite(market_close) & market_close.gt(0)
        )
        market_returns = market_close.pct_change(fill_method=None)
        market_var = market_returns.rolling(60, min_periods=60).var()
        frames = []
        for instrument, stock in quotes.groupby(self.instrument_col, sort=False):
            stock = stock.set_index(self.date_col).reindex(dates)

            def numeric(name):
                return pd.to_numeric(
                    stock.get(name, pd.Series(np.nan, index=dates)), errors="coerce"
                ).replace([np.inf, -np.inf], np.nan)

            close, adjustment = numeric("Close"), numeric("Adj_factor")
            price = (close * adjustment).where(close.gt(0) & adjustment.gt(0))
            returns = price.pct_change(fill_method=None)
            cap = numeric("Circ_mv")
            turnover = numeric("Turnover_rate_f")
            risk = pd.DataFrame(index=dates)
            risk["SIZE"] = np.log(cap.where(cap.gt(0)))
            risk["VOL20"] = returns.rolling(20, min_periods=20).std()
            risk["TURN20"] = (
                turnover.where(turnover.ge(0)).rolling(20, min_periods=20).mean()
            )
            risk["BETA60"] = returns.rolling(60, min_periods=60).cov(
                market_returns
            ) / market_var.where(market_var.gt(1e-12))
            risk["INDUSTRY"] = stock.get("Ind_class_l1", pd.Series(np.nan, index=dates))
            risk[self.instrument_col] = instrument
            risk.index.name = self.date_col
            frames.append(risk.reset_index())
        columns = keys + list(self.FACTORS) + ["INDUSTRY"]
        if not frames:
            return pd.DataFrame(columns=columns)
        result = pd.concat(frames, ignore_index=True)[columns]
        result[self.date_col] = result[self.date_col].dt.strftime("%Y-%m-%d")
        return result

    @staticmethod
    def _corr(x, y):
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
            return np.nan
        return x.corr(y, method="spearman")

    @staticmethod
    def _standardize(values):
        """Winsorize risks, then z-score on the signal pool, never on future labels."""
        clipped = values.clip(values.quantile(0.01), values.quantile(0.99))
        std = clipped.std(ddof=0)
        if not np.isfinite(std) or std <= 1e-12:
            return clipped * 0.0
        return (clipped - clipped.mean()) / std

    def _exposures(self, group, selected, risks):
        rows = []
        for factor in self.FACTORS:
            values = group[factor]
            observed = values.notna()
            rows.append(
                {
                    "Factor": factor,
                    "ScoreCorr": self._corr(group[self.pred_col], values),
                    "Exposure": risks.loc[selected, factor].mean()
                    - risks[factor].mean(),
                    "PoolCoverage": observed.mean(),
                    "TopCoverage": observed.loc[selected].mean(),
                }
            )
        industry = group["INDUSTRY"]
        for category in sorted(industry.dropna().unique(), key=str):
            values = industry.eq(category).astype(float).where(industry.notna())
            rows.append(
                {
                    "Factor": "INDUSTRY:" + str(category),
                    "ScoreCorr": self._corr(group[self.pred_col], values),
                    "Exposure": values.loc[selected].mean() - values.mean(),
                    "PoolCoverage": industry.notna().mean(),
                    "TopCoverage": industry.loc[selected].notna().mean(),
                }
            )
        return rows

    def _group_stats(self, group):
        labels = group[self.label_col]
        scores = group[self.pred_col]
        result = dict(
            Stocks=len(group),
            LabelCoverage=labels.notna().mean(),
            RankIC=np.nan,
            HighExcess=np.nan,
            HighLow=np.nan,
            HighLabelCoverage=np.nan,
            LowLabelCoverage=np.nan,
        )
        if len(group) < self.min_group_size:
            return result
        result["RankIC"] = self._corr(scores, labels)
        low, high = scores.quantile([0.2, 0.8])
        if low >= high:
            return result
        # Keep tied scores together; select before examining future labels.
        high_labels, low_labels = labels[scores >= high], labels[scores <= low]
        result.update(
            HighExcess=high_labels.mean() - labels.mean(),
            HighLow=high_labels.mean() - low_labels.mean(),
            HighLabelCoverage=high_labels.notna().mean(),
            LowLabelCoverage=low_labels.notna().mean(),
        )
        return result

    def _groups(self, group):
        rows = []
        for factor in ("VOL20", "SIZE", "INDUSTRY"):
            values = group[factor]
            if factor == "INDUSTRY":
                buckets = values
            elif values.nunique() > 1:
                # Duplicate boundaries collapse; identical risks are never split.
                buckets = pd.qcut(values, 5, labels=False, duplicates="drop") + 1
            else:
                continue
            for bucket, members in group.groupby(buckets, sort=True):
                label = str(bucket) if factor == "INDUSTRY" else str(int(bucket))
                rows.append(
                    dict(Factor=factor, Group=label, **self._group_stats(members))
                )
        return rows

    def _attribution(self, group, selected, risks):
        valid = (
            risks.notna().all(axis=1)
            & group["INDUSTRY"].notna()
            & group[self.label_col].notna()
        )
        pool = group.loc[valid]
        top = pool.index.isin(selected)
        result = dict(
            Status="insufficient_samples",
            Stocks=len(pool),
            TopStocks=int(top.sum()),
            PoolCoverage=valid.mean(),
            TopCoverage=valid.loc[selected].mean(),
            PoolExcess=np.nan,
            RawRankIC=np.nan,
            ResidualRankIC=np.nan,
            ReconstructionError=np.nan,
            **{name: np.nan for name in self.CONTRIBUTIONS},
        )
        if len(pool) < max(self.min_regression_size, 2 * self.top_k):
            return result
        if not top.any():
            result["Status"] = "no_observed_top"
            return result
        industry = pd.get_dummies(pool["INDUSTRY"], drop_first=True, dtype=float)
        z = risks.loc[valid]
        active = [name for name in self.FACTORS if z[name].std(ddof=0) > 1e-12]
        x = np.column_stack(
            [np.ones(len(pool)), industry.to_numpy(), z[active].to_numpy()]
        )
        y = pool[self.label_col].to_numpy(dtype=float)
        coef, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
        if rank < x.shape[1] or len(pool) <= rank:
            result["Status"] = "rank_deficient"
            return result
        contributions = (x[top].mean(axis=0) - x.mean(axis=0)) * coef
        residual = y - x @ coef
        offset = 1 + industry.shape[1]
        result.update(
            Status="ok",
            PoolExcess=y[top].mean() - y.mean(),
            Industry=contributions[1:offset].sum(),
            Residual=residual[top].mean() - residual.mean(),
            RawRankIC=self._corr(pool[self.pred_col], pool[self.label_col]),
            ResidualRankIC=(
                self._corr(pool[self.pred_col], pd.Series(residual, index=pool.index))
                if residual.std() > 1e-12
                else np.nan
            ),
        )
        result.update({name: 0.0 for name in self.FACTORS})
        result.update(dict(zip(active, contributions[offset:])))
        result["ReconstructionError"] = result["PoolExcess"] - sum(
            result[name] for name in self.CONTRIBUTIONS
        )
        return result

    def evaluate(self, df):
        """Return daily numeric tables for the current universe's finite scores.

        Select TopK before excluding missing risks or labels. Attribution uses
        complete cases on both sides of the comparison and reports coverage.
        The industry/factor coefficients are explanatory, not live forecasts.
        """
        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col]).dt.strftime("%Y-%m-%d")
        if df.duplicated([self.date_col, self.instrument_col]).any():
            raise ValueError("Risk evaluation requires unique stock/date keys")
        for name in (self.pred_col, self.label_col, *self.FACTORS):
            df[name] = pd.to_numeric(df[name], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
        df = df.loc[df[self.pred_col].notna()]
        if "IN_UNIVERSE" in df:
            df = df.loc[df["IN_UNIVERSE"].eq(True)]
        tables = {"exposures": [], "groups": [], "attribution": []}
        for date, group in df.groupby(self.date_col, sort=True):
            group = group.reset_index(drop=True)
            selected = (
                group.sort_values(
                    [self.pred_col, self.instrument_col],
                    ascending=[False, True],
                    kind="stable",
                )
                .head(self.top_k)
                .index
            )
            risks = group[list(self.FACTORS)].apply(self._standardize)
            for name, rows in (
                ("exposures", self._exposures(group, selected, risks)),
                ("groups", self._groups(group)),
                ("attribution", [self._attribution(group, selected, risks)]),
            ):
                tables[name].extend({self.date_col: date, **row} for row in rows)
        return {name: pd.DataFrame(rows) for name, rows in tables.items()}

    def summarize(self, tables, period=None):
        """Equal-date averages; period may be None, 'M' or 'Y'. Keep daily tables.

        Days and metric-specific counts prevent skipped dates from appearing to
        be zero returns. Coverage is averaged over all attempted dates.
        """
        summaries = {}
        for name, table in tables.items():
            if table.empty:
                summaries[name] = pd.DataFrame()
                continue
            table = table.copy()
            table["Period"] = (
                pd.to_datetime(table[self.date_col]).dt.to_period(period).astype(str)
                if period
                else "All"
            )
            keys = ["Period"] + (
                {"exposures": ["Factor"], "groups": ["Factor", "Group"]}.get(name, [])
            )
            grouped = table.groupby(keys, sort=True)
            numeric = table.select_dtypes(include=np.number).columns
            summary = grouped[list(numeric)].mean()
            summary["Days"] = grouped.size()
            count_cols = {
                "exposures": ["Exposure", "ScoreCorr"],
                "groups": ["RankIC", "HighExcess", "HighLow"],
                "attribution": ["PoolExcess", "RawRankIC", "ResidualRankIC"],
            }[name]
            for column in count_cols:
                summary[column + "Days"] = grouped[column].count()
            summaries[name] = summary.reset_index()
        return summaries

    def format_report(self, tables):
        summary = self.summarize(tables)
        yearly = self.summarize(tables, period="Y")

        def render(frame, columns, percentages=()):
            if frame.empty:
                return "无可评估样本。"
            frame = frame[columns].copy()
            for column in columns:
                if column in ("Period", "Factor", "Group"):
                    continue
                if column.endswith("Days"):
                    frame[column] = frame[column].map(lambda x: str(int(x)))
                else:
                    frame[column] = frame[column].map(
                        lambda x: (
                            "--"
                            if not np.isfinite(x)
                            else (f"{x:.2%}" if column in percentages else f"{x:.4f}")
                        )
                    )
            return frame.to_markdown(index=False, disable_numparse=True)

        exposures = render(
            summary["exposures"],
            [
                "Factor",
                "ScoreCorr",
                "Exposure",
                "PoolCoverage",
                "TopCoverage",
                "ExposureDays",
                "ScoreCorrDays",
            ],
            ("PoolCoverage", "TopCoverage"),
        )
        groups = render(
            summary["groups"],
            [
                "Factor",
                "Group",
                "Stocks",
                "RankIC",
                "HighExcess",
                "HighLow",
                "LabelCoverage",
                "HighLabelCoverage",
                "LowLabelCoverage",
                "Days",
                "RankICDays",
                "HighExcessDays",
                "HighLowDays",
            ],
            (
                "HighExcess",
                "HighLow",
                "LabelCoverage",
                "HighLabelCoverage",
                "LowLabelCoverage",
            ),
        )
        attribution_columns = [
            "Period",
            "PoolExcess",
            *self.CONTRIBUTIONS,
            "RawRankIC",
            "ResidualRankIC",
            "PoolCoverage",
            "TopCoverage",
            "Days",
            "PoolExcessDays",
        ]
        attribution = render(
            yearly["attribution"],
            attribution_columns,
            ("PoolExcess", *self.CONTRIBUTIONS, "PoolCoverage", "TopCoverage"),
        )
        status = tables["attribution"]
        status_text = (
            ", ".join(
                f"{name}={count}"
                for name, count in status["Status"].value_counts().items()
            )
            if not status.empty
            else "无样本"
        )
        return (
            f"预测风险暴露（连续变量为标准差单位；行业为权重差的小数）\n{exposures}\n\n"
            f"同风险组内评估（分位组1最低；组内高/低分各20%，同分并列）\n{groups}\n\n"
            f"Top{self.top_k} 收益归因（按年汇总每日均值，未扣费、未年化）\n{attribution}\n\n"
            f"归因日期状态：{status_text}\n"
            "归因在风险和标签完整的共同样本上计算，可能与全池 PoolExcess 不同；"
            "TopK 先选后过滤，缺失不补选。Pool/TopCoverage 在暴露表指风险覆盖率，"
            "在归因表指风险及标签共同覆盖率。\n"
            "行业贡献依赖所选风险模型，Residual 不是纯 alpha；本报告不归因实际持仓收益，"
            "不对重叠标签报告独立样本显著性。"
        )
