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
        feature_names=data_handler.feature_names,
        mode="train",
    )

    for i in range(len(train_dataset)):
        data_dict = train_dataset[i]
        print(data_dict["features"].shape, data_dict["labels"].shape)
