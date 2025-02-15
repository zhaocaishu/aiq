import numpy as np
from scipy.stats import spearmanr


class Evaluator:
    def __init__(self, pred_col="PRED", label_col="LABEL"):
        self.pred_col = pred_col
        self.label_col = label_col

    def evaluate(self, df):
        def compute_ic(group):
            return spearmanr(group[self.pred_col], group[self.label_col])[0]

        daily_ic = df.reset_index().groupby("Date").apply(compute_ic)

        metrics = {
            "IC": np.mean(daily_ic),
            "ICIR": np.mean(daily_ic) / np.std(daily_ic),
        }
        return metrics
