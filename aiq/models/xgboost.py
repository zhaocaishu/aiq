import xgboost as xgb
import pandas as pd

from aiq.dataset import Dataset

from .base import BaseModel


class XGBModel(BaseModel):
    """XGBModel Model"""

    def __init__(self, feature_cols, label_col, model_params):
        self.feature_cols = feature_cols
        self.label_col = label_col

        self.model_params = model_params

        self.model = None

    def fit(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset=None,
        early_stopping_rounds=50,
        verbose_eval=20,
        eval_results=dict()
    ):
        train_df = train_dataset.to_dataframe()
        x_train, y_train = train_df[self.feature_cols].values, train_df[self.label_col].values
        dtrain = xgb.DMatrix(x_train, label=y_train)
        evals = [(dtrain, "train")]

        if val_dataset is not None:
            valid_df = val_dataset.to_dataframe()
            x_valid, y_valid = valid_df[self.feature_cols].values, valid_df[self.label_col].values
            dvalid = xgb.DMatrix(x_valid, label=y_valid)
            evals.append((dvalid, "valid"))

        self.model = xgb.train(
            self.model_params,
            dtrain=dtrain,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
            evals_result=eval_results
        )
        eval_results["train"] = list(eval_results["train"].values())[0]
        if val_dataset is not None:
            eval_results["valid"] = list(eval_results["valid"].values())[0]

    def predict(self, dataset: Dataset):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.to_dataframe()[self.feature_cols].values
        predict_result = self.model.predict(xgb.DMatrix(x_test))
        dataset.add_column('prediction', predict_result)
        return dataset

    def get_feature_importance(self, *args, **kwargs) -> pd.Series:
        """get feature importance
        Notes
        -------
            parameters reference:
                https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.get_score
        """
        return pd.Series(self.model.get_score(*args, **kwargs)).sort_values(ascending=False)
