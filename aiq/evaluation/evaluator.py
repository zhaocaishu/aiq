import numpy as np
from scipy.stats import spearmanr


class Evaluator:
    def __init__(self, pred_col="PRED", label_col="LABEL"):
        self.pred_col = pred_col
        self.label_col = label_col

    def compute_r2(self, df):
        label = df[self.label_col].values
        pred = df[self.pred_col].values

        r2 = 1 - ((label - pred) ** 2).sum() / (label ** 2).sum()
        return r2

    def evaluate(self, df):
        def compute_ic(group):
            return spearmanr(group[self.pred_col], group[self.label_col])[0]

        daily_ic = df.reset_index().groupby("Date").apply(compute_ic)
        r2 = self.compute_r2(df)

        metrics = {
            "IC": np.mean(daily_ic),
            "ICIR": np.mean(daily_ic) / np.std(daily_ic),
            "R2": r2
        }
        return metrics
