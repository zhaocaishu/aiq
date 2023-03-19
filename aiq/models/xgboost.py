import os
import json
from typing import Tuple

import numpy as np
import xgboost as xgb
import pandas as pd
from scipy.stats import pearsonr

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
        eval_results=dict()
    ):
        train_df = train_dataset.to_dataframe()
        x_train, y_train = train_df[self.feature_cols_].values, train_df[self.label_col_].values
        dtrain = xgb.DMatrix(x_train, label=y_train)
        evals = [(dtrain, "train")]

        if val_dataset is not None:
            valid_df = val_dataset.to_dataframe()
            x_valid, y_valid = valid_df[self.feature_cols_].values, valid_df[self.label_col_].values
            dvalid = xgb.DMatrix(x_valid, label=y_valid)
            evals.append((dvalid, "valid"))

        self.model = xgb.train(
            self.model_params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
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
        x_test = dataset.to_dataframe()[self.feature_cols_].values
        predict_result = self.model.predict(xgb.DMatrix(x_test))
        dataset.add_column('PREDICTION', predict_result)
        return dataset

    def get_feature_importance(self, *args, **kwargs) -> pd.Series:
        """get feature importance
        Notes
        -------
            parameters reference:
                https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.get_score
        """
        return pd.Series(self.model.get_score(*args, **kwargs)).sort_values(ascending=False)

    def save(self, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_file = os.path.join(model_dir, 'model.json')
        self.model.save_model(model_file)

        model_params = {
            'feature_cols': self.feature_cols_,
            'label_col': self.label_col_,
            'model_params': self.model_params
        }
        with open(os.path.join(model_dir, 'model.params'), 'w') as f:
            json.dump(model_params, f)

    def load(self, model_dir):
        self.model = xgb.Booster(model_file=os.path.join(model_dir, 'model.json'))
        with open(os.path.join(model_dir, 'model.params'), 'r') as f:
            model_params = json.load(f)
            self.feature_cols_ = model_params['feature_cols']
            self.label_col_ = model_params['label_col']
            self.model_params = model_params['model_params']
