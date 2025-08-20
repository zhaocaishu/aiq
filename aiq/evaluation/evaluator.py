import numpy as np
import pandas as pd
from scipy.stats import spearmanr


class Evaluator:
    def __init__(self, pred_col="PRED", label_col="LABEL", min_samples=5):
        """
        初始化 Evaluator 类，用于评估预测模型的性能。

        参数:
            pred_col (str): 预测值列名，默认为 'PRED'。
            label_col (str): 标签值列名，默认为 'LABEL'。
            min_samples (int): 计算斯皮尔曼相关系数的最小样本量，默认为 5。
        """
        self.pred_col = pred_col
        self.label_col = label_col
        self.min_samples = min_samples

    def compute_r2(self, df):
        """
        计算样本外 R²（决定系数）。

        参数:
            df (pandas.DataFrame): 包含预测值和标签的 DataFrame。

        返回:
            float: R² 值，若分母为零或数据无效则返回 NaN。
        """
        if not all(col in df.columns for col in [self.pred_col, self.label_col]):
            raise ValueError(
                f"DataFrame must contain columns: {self.pred_col}, {self.label_col}"
            )

        label = df[self.label_col].values
        pred = df[self.pred_col].values

        if len(label) == 0 or np.isnan(label).any() or np.isnan(pred).any():
            return np.nan

        sse = np.sum((label - pred) ** 2)
        sst = np.sum(label**2)
        return 1 - sse / sst if sst != 0 else np.nan

    def compute_ic(self, group):
        """
        计算单组数据的斯皮尔曼相关系数 (IC)。

        参数:
            group (pandas.DataFrame): 单日数据，包含预测值和标签。

        返回:
            float: 斯皮尔曼相关系数，若样本不足或数据无效则返回 NaN。
        """
        if (
            len(group) < self.min_samples
            or group[self.pred_col].isna().any()
            or group[self.label_col].isna().any()
        ):
            return np.nan
        return spearmanr(group[self.pred_col], group[self.label_col])[0]

    def compute_ic_top_bottom(self, group, top_ratio=0.3, bottom_ratio=0.3):
        """
        计算top X%和bottom Y%样本的IC。

        参数:
            group (pd.DataFrame): 单日数据，包含预测值和标签。
            top_ratio (float): Top 部分比例，默认 0.3。
            bottom_ratio (float): Bottom 部分比例，默认 0.3。

        返回:
            dict: {"IC_Top": float, "IC_Bottom": float}
        """
        n = len(group)
        if n < self.min_samples:
            return {"IC_Top": np.nan, "IC_Bottom": np.nan}

        group_sorted = group.sort_values(by=self.pred_col, ascending=False)

        top_n = max(int(n * top_ratio), self.min_samples)
        top_group = group_sorted.head(top_n)
        ic_top = (
            spearmanr(top_group[self.pred_col], top_group[self.label_col])[0]
            if len(top_group) >= self.min_samples
            else np.nan
        )

        bottom_n = max(int(n * bottom_ratio), self.min_samples)
        bottom_group = group_sorted.tail(bottom_n)
        ic_bottom = (
            spearmanr(bottom_group[self.pred_col], bottom_group[self.label_col])[0]
            if len(bottom_group) >= self.min_samples
            else np.nan
        )

        return {"IC_Top": ic_top, "IC_Bottom": ic_bottom}

    def compute_hit_rate(self, group, K=30):
        """
        计算 Top-K 和 Bottom-K 的命中率（Hit Rate）。

        参数:
            group (pd.DataFrame): 单日数据，包含预测值列和真实标签列。
            K (int): Top-K 和 Bottom-K 的取值。

        返回:
            pd.DataFrame: 包含 HR@TopK 和 HR@BottomK 的 DataFrame。若样本不足则返回 NaN。
        """
        if len(group) < K or K <= 0:
            return pd.DataFrame({f"HR@Top{K}": [np.nan], f"HR@Bottom{K}": [np.nan]})

        # Top-K
        pred_topk = set(group.nlargest(K, self.pred_col)["Instrument"])
        gt_topk = set(group.nlargest(K, self.label_col)["Instrument"])
        hr_top = len(pred_topk & gt_topk) / K

        # Bottom-K
        pred_bottomk = set(group.nsmallest(K, self.pred_col)["Instrument"])
        gt_bottomk = set(group.nsmallest(K, self.label_col)["Instrument"])
        hr_bottom = len(pred_bottomk & gt_bottomk) / K

        return pd.DataFrame({f"HR@Top{K}": [hr_top], f"HR@Bottom{K}": [hr_bottom]})

    def calc_spread_return_sharpe(
        self,
        df: pd.DataFrame,
        portfolio_size: int = 30,
        toprank_weight_ratio: float = 2,
    ) -> float:
        """
        计算基于 spread return 的 Sharpe 比率。

        参数:
            df (pd.DataFrame): 预测结果 DataFrame。
            portfolio_size (int): 买卖的股票数量。
            toprank_weight_ratio (float): 最高排名股票相对于最低的权重比例。

        返回:
            float: Sharpe 比率。
        """

        def _calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):
            assert df["Rank"].min() == 0
            assert df["Rank"].max() == len(df["Rank"]) - 1
            weights = np.linspace(
                start=toprank_weight_ratio, stop=1, num=portfolio_size
            )
            purchase = (
                df.sort_values(by="Rank")[self.label_col][:portfolio_size] * weights
            ).sum() / weights.mean()
            short = (
                df.sort_values(by="Rank", ascending=False)[self.label_col][
                    :portfolio_size
                ]
                * weights
            ).sum() / weights.mean()
            return purchase - short

        df["Rank"] = df[self.pred_col].rank(method="dense", ascending=False) - 1
        buf = df.groupby("Date").apply(
            _calc_spread_return_per_day, portfolio_size, toprank_weight_ratio
        )
        return buf.mean() / buf.std() if not buf.empty else np.nan

    def evaluate(self, df, groupby_col="Date"):
        """
        评估预测模型性能，返回 IC、ICIR、R² 等指标。

        参数:
            df (pandas.DataFrame): 包含日期、预测值和标签的 DataFrame。
            groupby_col (str): 分组列名，默认为 'Date'。

        返回:
            dict: 包含 IC、ICIR、R2、HitRate、SpreadReturn 等指标的字典。
        """
        if groupby_col not in df.columns:
            raise ValueError(f"DataFrame must contain groupby column: {groupby_col}")

        # 计算每日 IC
        daily_ic = df.groupby(groupby_col).apply(self.compute_ic).dropna()
        ic_mean = daily_ic.mean() if not daily_ic.empty else np.nan
        ic_std = daily_ic.std() if not daily_ic.empty else np.nan
        icir = ic_mean / ic_std if ic_std != 0 else np.nan

        # 计算 top30% 和 bottom30% 的 IC
        daily_ic_tb = df.groupby(groupby_col).apply(self.compute_ic_top_bottom)
        ic_top_mean = (
            pd.DataFrame(daily_ic_tb.tolist())["IC_Top"].mean()
            if not daily_ic_tb.empty
            else np.nan
        )
        ic_bottom_mean = (
            pd.DataFrame(daily_ic_tb.tolist())["IC_Bottom"].mean()
            if not daily_ic_tb.empty
            else np.nan
        )

        # 计算 R²
        r2 = self.compute_r2(df)

        # 计算命中率
        daily_hitrate = df.groupby(groupby_col).apply(self.compute_hit_rate)
        hitrate = daily_hitrate.mean().to_dict()

        # 计算超额收益
        spread_return = self.calc_spread_return_sharpe(df)

        return {
            "IC": ic_mean,
            "ICIR": icir,
            "IC_Top30%": ic_top_mean,
            "IC_Bottom30%": ic_bottom_mean,
            "R2": r2,
            "HitRate": hitrate,
            "SpreadReturn": spread_return,
        }
