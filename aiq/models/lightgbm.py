import os
import json

import lightgbm as lgb
import pandas as pd

from aiq.dataset import Dataset

from .base import BaseModel


class LGBModel(BaseModel):
    """LGBModel Model"""

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
        dtrain = lgb.Dataset(x_train, label=y_train)
        evals = [dtrain]

        if val_dataset is not None:
            valid_df = val_dataset.data
            x_valid, y_valid = (
                valid_df[self._feature_cols].values,
                valid_df[self._label_cols].values,
            )
            dvalid = lgb.Dataset(x_valid, label=y_valid)
            evals.append(dvalid)

        early_stopping_callback = lgb.early_stopping(
            self.early_stopping_rounds
            if early_stopping_rounds is None
            else early_stopping_rounds
        )
        # NOTE: if you encounter error here. Please upgrade your lightgbm
        verbose_eval_callback = lgb.log_evaluation(period=verbose_eval)
        evals_result_callback = lgb.record_evaluation(eval_results)

        self.model = lgb.train(
            self.model_params,
            train_set=dtrain,
            num_boost_round=(
                self.num_boost_round if num_boost_round is None else num_boost_round
            ),
            valid_sets=evals,
            valid_names=["train", "valid"],
            callbacks=[
                early_stopping_callback,
                verbose_eval_callback,
                evals_result_callback,
            ],
        )

    def predict(self, test_dataset: Dataset):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test = test_dataset.data[self._feature_cols].values
        preds = self.model.predict(x_test)
        test_dataset.insert(cols=["PRED"], data=preds)
        return test_dataset

    def get_feature_importance(self, *args, **kwargs) -> pd.Series:
        """get feature importance
        Notes
        -------
            parameters reference:
                https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.get_score
        """
        return pd.Series(self.model.feature_importance(*args, **kwargs)).sort_values(
            ascending=False
        )

    def save(self, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_file = os.path.join(model_dir, "model.json")
        self.model.save_model(model_file)

    def load(self, model_dir):
        self.model = lgb.Booster(model_file=os.path.join(model_dir, "model.json"))
