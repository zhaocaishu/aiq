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

        # 检查数据是否有效
        if len(label) == 0 or np.any(np.isnan(label)) or np.any(np.isnan(pred)):
            return np.nan

        # 计算残差平方和 (SSE)
        sse = np.sum((label - pred) ** 2)

        # 计算总平方和 (SST)
        sst = np.sum((label - label.mean()) ** 2)

        # 避免除以零
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

    def compute_hit_rate(self, group, K=20):
        """
        计算 Top-K 和 Bottom-K 的命中率（Hit Rate）。

        参数:
            group (pd.DataFrame): 单日数据，包含预测值列和真实标签列。
            K (int): Top-K 和 Bottom-K 的取值。

        返回:
            Tuple[float, float]: 返回 Top-K 和 Bottom-K 的命中率 (HR@TopK, HR@BottomK)。
                                若样本数量不足或数据无效，则返回 (np.nan, np.nan)。
        """
        # 样本量不足时直接返回 NaN
        if (
            group is None
            or not isinstance(group, pd.DataFrame)
            or len(group) < K
            or K <= 0
        ):
            return pd.DataFrame({"HR@TopK": [np.nan], "HR@BottomK": [np.nan]})

        # 排序并取 Top-K
        topk = group.nlargest(K, self.pred_col)
        # 排序并取 Bottom-K
        bottomk = group.nsmallest(K, self.pred_col)

        # 计算命中数（标签 == 1），并除以 K 得到命中率
        hr_top = topk[self.label_col].eq(1).sum() / K
        hr_bottom = bottomk[self.label_col].eq(1).sum() / K

        return pd.DataFrame({"HR@TopK": [hr_top], "HR@BottomK": [hr_bottom]})

    def evaluate(self, df, groupby_col="Date"):
        """
        评估预测模型性能，返回 IC、ICIR 和 R² 等指标。

        参数:
            df (pandas.DataFrame): 包含日期、预测值和标签的 DataFrame。
            groupby_col (str): 分组列名，默认为 'Date'。

        返回:
            dict: 包含以下指标的字典：
                - IC: 每日斯皮尔曼相关系数的均值。
                - ICIR: IC 的均值除以标准差。
                - R2: 样本外 R²。
                - 命中率: Top-K 和 Bottom-K 的命中率。
        """
        if not groupby_col in df.columns:
            raise ValueError(f"DataFrame must contain groupby column: {groupby_col}")

        # 计算每日 IC
        daily_ic = df.groupby(groupby_col).apply(self.compute_ic)

        # 过滤 NaN 并计算 IC 和 ICIR
        daily_ic = daily_ic.dropna()
        ic_mean = daily_ic.mean() if not daily_ic.empty else np.nan
        ic_std = daily_ic.std() if not daily_ic.empty else np.nan
        icir = ic_mean / ic_std if ic_std != 0 else np.nan

        # 计算 R²
        r2 = self.compute_r2(df)

        # 计算命中率
        daily_hitrate = df.groupby(groupby_col).apply(self.compute_hit_rate)
        hitrate = daily_hitrate.mean()

        metrics = {"IC": ic_mean, "ICIR": icir, "R2": r2, "HitRate": hitrate.to_dict()}
        return metrics
