import os

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
        eval_results=None,
    ):
        if eval_results is None:
            eval_results = {}
        x_train, y_train = train_dataset.get_training_data(
            self.feature_names, self.label_names
        )
        dtrain = lgb.Dataset(x_train, label=y_train.ravel())
        evals = [dtrain]
        valid_names = ["train"]
        callbacks = [
            lgb.log_evaluation(period=verbose_eval),
            lgb.record_evaluation(eval_results),
        ]

        if val_dataset is not None:
            x_valid, y_valid = val_dataset.get_training_data(
                self.feature_names, self.label_names
            )
            dvalid = lgb.Dataset(x_valid, label=y_valid.ravel())
            evals.append(dvalid)
            valid_names.append("valid")
            if early_stopping_rounds:
                callbacks.append(lgb.early_stopping(early_stopping_rounds))

        self.model = lgb.train(
            self.model_params,
            train_set=dtrain,
            num_boost_round=num_boost_round,
            valid_sets=evals,
            valid_names=valid_names,
            callbacks=callbacks,
        )

    def predict(self, test_dataset: Dataset):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test = test_dataset.data[self.feature_names].values
        preds = self.model.predict(x_test)
        pred_df = test_dataset.data.copy(deep=False)
        pred_df[f"PRED_{self.label_names[0]}"] = preds
        return pred_df

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

    def load(self, model_name=None):
        model_name = "model.json" if model_name is None else model_name
        self.model = lgb.Booster(model_file=os.path.join(self.save_dir, model_name))

    def save(self, model_name=None):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        model_name = "model.json" if model_name is None else model_name
        model_file = os.path.join(self.save_dir, model_name)
        self.model.save_model(model_file)
