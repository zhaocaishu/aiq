from aiq.utils.config import config as cfg
from aiq.utils.module import init_instance_by_config


if __name__ == "__main__":
    # config
    cfg.from_file("./configs/xgboost_model_reg.yaml")

    # data handler
    data_handler = init_instance_by_config(cfg.data_handler, data_dir="./data")
    data = data_handler.setup_data()

    # train and validation dataset
    train_dataset = init_instance_by_config(
        cfg.dataset,
        data=data,
        daily_instruments=data_handler.daily_instruments,
        feature_names=data_handler.feature_names,
        mode="train",
    )

    val_dataset = init_instance_by_config(
        cfg.dataset,
        data=data,
        daily_instruments=data_handler.daily_instruments,
        feature_names=data_handler.feature_names,
        mode="valid",
    )

    print(train_dataset.data.shape, val_dataset.data.shape)
