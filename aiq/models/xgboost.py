import os

import xgboost as xgb
import pandas as pd

from aiq.dataset import Dataset

from .base import BaseModel


class XGBModel(BaseModel):
    """XGBModel Model"""

    def fit(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset = None,
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose_eval=20,
        eval_results=dict(),
    ):
        train_df = train_dataset.data
        x_train, y_train = (
            train_df[self._feature_cols].values,
            train_df[self._label_cols].values,
        )
        dtrain = xgb.DMatrix(x_train, label=y_train)
        evals = [(dtrain, "train")]

        if val_dataset is not None:
            valid_df = val_dataset.data
            x_valid, y_valid = (
                valid_df[self._feature_cols].values,
                valid_df[self._label_cols].values,
            )
            dvalid = xgb.DMatrix(x_valid, label=y_valid)
            evals.append((dvalid, "valid"))

        self.model = xgb.train(
            self.model_params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
            evals_result=eval_results,
        )
        eval_results["train"] = list(eval_results["train"].values())[0]
        if val_dataset is not None:
            eval_results["valid"] = list(eval_results["valid"].values())[0]

    def predict(self, test_dataset: Dataset):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        test_df = test_dataset.data[self._feature_cols]
        dtest = xgb.DMatrix(test_df.values)
        preds = self.model.predict(dtest)
        test_dataset.insert(cols=["PRED"], data=preds)
        return test_dataset

    def get_feature_importance(self, *args, **kwargs) -> pd.Series:
        """get feature importance
        Notes
        -------
            parameters reference:
                https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.get_score
        """
        return pd.Series(self.model.get_score(*args, **kwargs)).sort_values(
            ascending=False
        )

    def save(self, model_name=None):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        model_name = "model.json" if model_name is None else model_name
        model_file = os.path.join(self.save_dir, model_name)
        self.model.save_model(model_file)

    def load(self, model_name=None):
        model_name = "model.json" if model_name is None else model_name
        self.model = xgb.Booster(model_file=os.path.join(self.save_dir, model_name))
