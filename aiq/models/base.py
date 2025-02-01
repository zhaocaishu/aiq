import abc
import copy

from torch.utils.data import Dataset


class BaseModel(abc.ABC):
    """Learnable Models"""

    def __init__(
        self, feature_cols=None, label_col=None, model_params=None, logger=None
    ):
        self._feature_cols = feature_cols
        self._label_col = label_col

        if model_params is not None:
            self.model_params = copy.deepcopy(dict(model_params))
        else:
            self.model_params = {}

        self.model = None

        self.logger = logger

    def fit(self, train_dataset: Dataset, val_dataset: Dataset = None):
        """
        Learn model from the base model

        Args:
            train_dataset: train dataset
            val_dataset: validation dataset

        Returns:
            Trained model
        """
        raise NotImplementedError()

    def predict(self, dataset: Dataset) -> object:
        """
        Give prediction given Dataset

        Args:
            dataset:  dataset will generate the processed dataset from model training.

        Returns:
             Prediction results with certain type such as `pandas.Series`.
        """
        raise NotImplementedError()

    @property
    def feature_cols(self):
        return self._feature_cols

    @property
    def label_col(self):
        return self._label_col
