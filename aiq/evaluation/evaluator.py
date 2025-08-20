import numpy as np
import pandas as pd
from scipy.stats import spearmanr


class Evaluator:
    def __init__(self, pred_col="PRED", label_col="LABEL", min_samples=5):
        """
        预测模型评估工具类
        """
        self.pred_col = pred_col
        self.label_col = label_col
        self.min_samples = min_samples

    def _check_columns(self, df, extra_cols=None):
        """检查 DataFrame 是否包含必要的列"""
        required = {self.pred_col, self.label_col}
        if extra_cols:
            required |= set(extra_cols)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def compute_r2(self, df):
        """样本外 R²（决定系数）"""
        self._check_columns(df)
        label, pred = df[self.label_col].values, df[self.pred_col].values
        if len(label) == 0 or np.isnan(label).any() or np.isnan(pred).any():
            return np.nan
        sse = np.sum((label - pred) ** 2)
        sst = np.sum(label**2)
        return 1 - sse / sst if sst != 0 else np.nan

    def compute_ic(self, group):
        """单组斯皮尔曼相关系数 (IC)"""
        if len(group) < self.min_samples:
            return np.nan
        return spearmanr(group[self.pred_col], group[self.label_col])[0]

    def compute_ic_top_bottom(self, group, K=30):
        """计算 Top/Bottom 部分的 IC"""
        n = len(group)
        if n < self.min_samples:
            return {f"IC@Top{K}": np.nan, f"IC@Bottom{K}": np.nan}

        group_sorted = group.sort_values(by=self.pred_col, ascending=False)

        def _sub_ic(subset):
            return spearmanr(subset[self.pred_col], subset[self.label_col])[0]

        ic_top = _sub_ic(group_sorted.head(K))
        ic_bottom = _sub_ic(group_sorted.tail(K))

        return {f"IC@Top{K}": ic_top, f"IC@Bottom{K}": ic_bottom}

    def compute_hit_rate(self, group, K=30):
        """Top-K 和 Bottom-K 命中率"""
        self._check_columns(group, extra_cols=["Instrument"])
        if len(group) < K or K <= 0:
            return {f"HR@Top{K}": np.nan, f"HR@Bottom{K}": np.nan}

        pred_top = set(group.nlargest(K, self.pred_col)["Instrument"])
        label_top = set(group.nlargest(K, self.label_col)["Instrument"])
        pred_bottom = set(group.nsmallest(K, self.pred_col)["Instrument"])
        label_bottom = set(group.nsmallest(K, self.label_col)["Instrument"])

        return {
            f"HR@Top{K}": len(pred_top & label_top) / K,
            f"HR@Bottom{K}": len(pred_bottom & label_bottom) / K,
        }

    def evaluate(self, df, groupby_col="Date"):
        """整体评估：IC、ICIR、R²、Top/Bottom IC、HitRate"""
        self._check_columns(df, extra_cols=[groupby_col, "Instrument"])

        # === IC & ICIR ===
        daily_ic = df.groupby(groupby_col).apply(self.compute_ic).dropna()
        ic_mean, ic_std = daily_ic.mean(), daily_ic.std()
        icir = ic_mean / ic_std if ic_std != 0 else np.nan

        # === Top/Bottom IC ===
        daily_ic_tb = df.groupby(groupby_col).apply(self.compute_ic_top_bottom)
        ic_tb_mean = pd.DataFrame(daily_ic_tb.tolist()).mean().to_dict()

        # === R² ===
        r2 = self.compute_r2(df)

        # === Hit Rate ===
        daily_hr = df.groupby(groupby_col).apply(self.compute_hit_rate)
        hr_mean = pd.DataFrame(daily_hr.tolist()).mean().to_dict()

        # === 汇总结果 ===
        results = {
            "IC": ic_mean,
            "ICIR": icir,
            "R2": r2,
        }
        results.update(ic_tb_mean)
        results.update(hr_mean)

        return pd.DataFrame([results]).to_markdown(index=False, floatfmt=".4f")
