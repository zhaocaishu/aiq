import abc
from typing import Union, Text

import pandas as pd

from aiq.utils.data import robust_zscore, zscore


def get_group_columns(df: pd.DataFrame, group: Union[Text, None]):
    """
    Get a group of columns from multi-index columns DataFrame

    Args:
        df (pd.DataFrame): With multi of columns.
        group (str): the name of the feature group, i.e. the first level value of the group index.
    """
    if group is None:
        return df.columns
    else:
        return df.columns[df.columns.get_loc(group)]


class Processor(abc.ABC):
    def fit(self, df: pd.DataFrame = None):
        """
        Learn data processing parameters

        Args:
            df (pd.DataFrame): When we fit and process data with processor one by one. The fit function reiles on the output of previous
            processor, i.e. `df`.

        """

    @abc.abstractmethod
    def __call__(self, df: pd.DataFrame):
        """
        Process the data
        NOTE: **The processor could change the content of `df` inplace !!!!! **
        User should keep a copy of data outside

        Args:
            df (pd.DataFrame): The raw_df of handler or result from previous processor.
        """


class CSZScoreNorm(Processor):
    """Cross Sectional ZScore Normalization"""

    def __init__(self, fields_group=None, method="zscore"):
        self.fields_group = fields_group
        if method == "zscore":
            self.zscore_func = zscore
        elif method == "robust":
            self.zscore_func = robust_zscore
        else:
            raise NotImplementedError(f"This type of input is not supported")

    def __call__(self, df):
        # try not modify original dataframe
        if not isinstance(self.fields_group, list):
            self.fields_group = [self.fields_group]
        for g in self.fields_group:
            cols = get_group_columns(df, g)
            df[cols] = df.groupby("Date", group_keys=False)[cols].apply(self.zscore_func)
        return df


class FeatureGroupMean(Processor):
    """Feature mean group by group id"""

    def __init__(self, fields_group=None, group_names=['Date', 'Industry_id']):
        self.fields_group = fields_group
        self.group_names = group_names

    def __call__(self, df):
        if not isinstance(self.fields_group, list):
            self.fields_group = [self.fields_group]

        feature_names = []
        for col in self.fields_group:
            df[f'M{col}'] = df.groupby(self.group_names)[col].transform('mean')
            feature_names.append(f'M{col}')
        return df, feature_names


class RandomLabelSampling(Processor):
    """Random sampling on label"""

    def __init__(self, label_name=None, bound_value=None, sample_ratio=0.5):
        self.label_name = label_name
        self.bound_value = bound_value
        self.sample_ratio = sample_ratio

    def __call__(self, df):
        df1 = df[(df[self.label_name] >= self.bound_value[0]) & (df[self.label_name] <= self.bound_value[1])]
        df2 = df[~df.index.isin(df1.index)]
        df1 = df1.sample(frac=self.sample_ratio)
        df = pd.concat([df1, df2])
        return df


class DropOutlierAndNorm(Processor):
    def __init__(self, feature_names=None, label_name=None, label_norm=False):
        self.feature_names = feature_names
        self.label_name = label_name
        self.label_norm = label_norm

    def _transform(self, df):
        def _feature_norm(x):
            x = x - x.median()  # copy
            x /= x.abs().median() * 1.4826
            x.clip(-3, 3, inplace=True)
            return x

        def _label_norm(x):
            if self.label_norm:
                x = x - x.mean()  # copy
                x /= x.std()
                x.clip(-3, 3, inplace=True)
            else:
                x.clip(-0.1, 0.1, inplace=True)
            return x

        # Feature
        df[self.feature_names] = df.groupby("Date", group_keys=False)[self.feature_names].apply(_feature_norm)

        # Label
        if self.label_name is not None:
            df[self.label_name] = df.groupby("Date", group_keys=False)[self.label_name].apply(_label_norm)

        return df

    def __call__(self, df):
        return self._transform(df)
