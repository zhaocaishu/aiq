from aiq.utils.config import config as cfg
from aiq.utils.module import init_instance_by_config
from aiq.dataset.dataset import MarketTSDataset


if __name__ == "__main__":
    # config
    cfg.from_file("./configs/matcc_model_reg.yaml")

    # data handler
    data_handler = init_instance_by_config(cfg.data_handler)

    # train dataset
    train_dataset = MarketTSDataset(
        data_dir="./data",
        instruments="csi300",
        segments=cfg.dataset.kwargs.segments,
        seq_len=8,
        pred_len=4,
        data_handler=data_handler,
        mode="train",
    )

    for i in range(len(train_dataset)):
        index, feature, label = train_dataset[i]
        print(index, feature.shape, label.shape)
