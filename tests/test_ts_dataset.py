from aiq.utils.config import config as cfg
from aiq.utils.module import init_instance_by_config
from aiq.dataset.dataset import MarketTSDataset


if __name__ == "__main__":
    # config
    cfg.from_file("./configs/ppnet_model_reg.yaml")

    # data handler
    data_handler = init_instance_by_config(cfg.data_handler)

    # train dataset
    train_dataset = MarketTSDataset(
        data_dir="./data",
        instruments="000300.SH",
        segments=cfg.dataset.kwargs.segments,
        seq_len=8,
        label_cols=["RETN_5D"],
        data_handler=data_handler,
        mode="train",
    )

    for i in range(len(train_dataset)):
        index, feature, label = train_dataset[i]
        print(index, feature.shape, label.shape)
