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
        instruments=["002750.SZ", "002811.SZ", "600490.SH"],
        start_time="2024-01-01",
        end_time="2024-04-30",
        data_handler=data_handler,
        mode="train",
    )
    
    # dataset
    val_dataset = Dataset(
        "./data",
        instruments=["002750.SZ", "002811.SZ", "600490.SH"],
        start_time="2024-01-01",
        end_time="2024-04-30",
        data_handler=data_handler,
        mode="valid",
    )
    print(val_dataset.to_dataframe().equals(train_dataset.to_dataframe()))
