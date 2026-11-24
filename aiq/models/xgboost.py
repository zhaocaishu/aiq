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
        eval_results=None,
    ):
        if eval_results is None:
            eval_results = {}
        x_train, y_train = train_dataset.get_training_data(
            self.feature_names, self.label_names
        )
        dtrain = xgb.DMatrix(x_train, label=y_train)
        evals = [(dtrain, "train")]

        if val_dataset is not None:
            x_valid, y_valid = val_dataset.get_training_data(
                self.feature_names, self.label_names
            )
            dvalid = xgb.DMatrix(x_valid, label=y_valid)
            evals.append((dvalid, "valid"))

        self.model = xgb.train(
            self.model_params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=(
                early_stopping_rounds if val_dataset is not None else None
            ),
            verbose_eval=verbose_eval,
            evals_result=eval_results,
        )
        eval_results["train"] = list(eval_results["train"].values())[0]
        if val_dataset is not None:
            eval_results["valid"] = list(eval_results["valid"].values())[0]

    def predict(self, test_dataset: Dataset):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        test_df = test_dataset.data[self.feature_names]
        dtest = xgb.DMatrix(test_df.values)
        # The native booster retains trees grown after the best validation round.
        best_iteration = self.model.attr("best_iteration")
        end = int(best_iteration) + 1 if best_iteration is not None else 0
        preds = self.model.predict(dtest, iteration_range=(0, end))
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
        return pd.Series(self.model.get_score(*args, **kwargs)).sort_values(
            ascending=False
        )

    def load(self, model_name=None):
        model_name = "model.json" if model_name is None else model_name
        self.model = xgb.Booster(model_file=os.path.join(self.save_dir, model_name))

    def save(self, model_name=None):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        model_name = "model.json" if model_name is None else model_name
        model_file = os.path.join(self.save_dir, model_name)
        self.model.save_model(model_file)
