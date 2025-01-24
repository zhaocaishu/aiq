from aiq.utils.config import config as cfg
from aiq.utils.module import init_instance_by_config
from aiq.dataset.dataset import TSDataset


if __name__ == "__main__":
    # config
    cfg.from_file("./configs/xgboost_model_reg.yaml")

    # data handler
    data_handler = init_instance_by_config(cfg.data_handler)

    # train dataset
    train_dataset = TSDataset(
        data_dir="./data",
        instruments="csi300",
        segments=cfg.dataset.kwargs.segments,
        seq_len=8,
        pred_len=1,
        data_handler=data_handler,
        mode="train",
    )

    for i in range(len(train_dataset)):
        date, feature, label = train_dataset[i]
        print(date, feature.shape, label.shape)
