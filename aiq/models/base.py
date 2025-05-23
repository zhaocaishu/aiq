import abc
import copy

from torch.utils.data import Dataset


class BaseModel(abc.ABC):
    """Learnable Models"""

    def __init__(
        self,
        feature_names=None,
        label_names=None,
        model_params=None,
        save_dir=None,
        logger=None,
    ):
        self._feature_names = feature_names
        self._label_names = label_names

        if model_params is not None:
            self.model_params = copy.deepcopy(dict(model_params))
        else:
            self.model_params = {}

        self.model = None

        self.save_dir = save_dir

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

    def predict(self, test_dataset: Dataset) -> object:
        """
        Give prediction given Dataset

        Args:
            test_dataset:  dataset for model prediction.

        Returns:
             Prediction results with certain type such as `pandas.Series`.
        """
        raise NotImplementedError()

    def load(self, model_name):
        """
        Loads object state from storage using the specified model name.

        Args:
            model_name: Unique identifier of the model to load from storage
        """
        raise NotImplementedError()

    def save(self, model_name):
        """
        Persistently saves the object's state to storage using the specified model name.

        Args:
            model_name: Unique identifier for the model in storage.
        """
        raise NotImplementedError()

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def label_names(self):
        return self._label_names
