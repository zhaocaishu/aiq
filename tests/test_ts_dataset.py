from aiq.utils.config import config as cfg
from aiq.utils.module import init_instance_by_config
from aiq.dataset.dataset import TSDataset


if __name__ == "__main__":
    # config
    cfg.from_file("./configs/ppnet_model_reg.yaml")

    # data handler
    data_handler = init_instance_by_config(cfg.data_handler, data_dir="./data")
    data = data_handler.setup_data()
    print(data_handler.feature_names)

    # train dataset
    train_dataset = TSDataset(
        data=data,
        segments=cfg.dataset.kwargs.segments,
        seq_len=8,
        data_dir="./data",
        universe=cfg.dataset.kwargs.universe,
        feature_names=data_handler.feature_names,
        label_names=["RETN_5D"],
        mode="train",
    )

    data_dict = train_dataset[0]
    print(
        data_dict["indices"].shape,
        data_dict["industries"].shape,
        data_dict["stock_features"].shape,
        data_dict["market_features"].shape,
        data_dict["labels"].shape,
    )
