import pandas as pd

from aiq.dataset import Dataset
from aiq.utils.config import config as cfg

if __name__ == "__main__":
    cfg.from_file("./configs/xgboost_model_reg.yaml")

    train_dataset = Dataset(
        "./data",
        instruments=["000951.SZ", "601099.SH", "688366.SH"],
        start_time="2012-01-01",
        end_time="2023-05-01",
        training=True,
    )
    print(train_dataset.to_dataframe())
