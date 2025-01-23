from aiq.utils.config import config as cfg
from aiq.utils.module import init_instance_by_config


if __name__ == "__main__":
    # config
    cfg.from_file("./configs/nlinear_model_reg.yaml")

    # data handler
    data_handler = init_instance_by_config(cfg.data_handler)

    # train and validation dataset
    train_dataset = init_instance_by_config(
        cfg.dataset,
        data_dir="./data",
        data_handler=data_handler,
        mode="train",
    )

    print(train_dataset)
