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
        if fields_group is None:
            self.fields_group = [
                'CNTN20', 'CNTN30', 'KSFT', 'CNTN5', 'MAX20', 'RSQR5', 'MAX30', 'HIGH0', 'RETURN60', 'RETURN20',
                'QTLU60', 'RESI30', 'RETURN30', 'KLOW', 'ROC30', 'LOW1', 'CNTP5', 'MAX5', 'HIGH1', 'RESI60', 'RETURN5',
                'MA20', 'HIGH2', 'IMIN60', 'OPEN1', 'KSFT2', 'ROC20', 'VOLUME1', 'CLOSE3', 'OPEN4', 'CLOSE4', 'OPEN3',
                'KLEN', 'RESI5', 'VOLAT60', 'VMA30', 'BETA60', 'MA5', 'ROC60', 'VSUMN5', 'QTLU20', 'STD5', 'CLOSE1',
                'MAX10', 'KUP', 'HIGH3', 'MA60', 'LOW0', 'BETA20', 'SUMP5'
            ]
        else:
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
