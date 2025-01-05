from aiq.dataset import inst_data_handler, Dataset
from aiq.utils.config import config as cfg

if __name__ == "__main__":
    cfg.from_file("./configs/xgboost_model_reg.yaml")

    # setup data handler
    data_handler_config = cfg["data_handler"]
    data_handler = inst_data_handler(data_handler_config)

    # dataset
    train_dataset = Dataset(
        "./data",
        instruments=["000951.SZ", "601099.SH", "688366.SH"],
        start_time="2012-01-01",
        end_time="2023-05-01",
        data_handler=data_handler,
        training=True,
    )
    print(train_dataset.to_dataframe())
