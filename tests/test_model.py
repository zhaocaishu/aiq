from aiq.models import LGBModel, MATCCModel
from aiq.utils.config import config as cfg
from aiq.utils.module import init_instance_by_config
from aiq.utils.logging import get_logger


if __name__ == "__main__":
    # config
    cfg.from_file("./configs/matcc_model_reg.yaml")

    # logger
    logger = get_logger("TEST_MODEL")

    # data handler
    data_handler = init_instance_by_config(cfg.data_handler)

    # train and validation dataset
    train_dataset = init_instance_by_config(
        cfg.dataset,
        data_dir="./data",
        data_handler=data_handler,
        mode="train",
    )

    val_dataset = init_instance_by_config(
        cfg.dataset,
        data_dir="./data",
        data_handler=data_handler,
        mode="valid",
    )

    print(train_dataset.feature_names, train_dataset.label_names)

    # train stage
    model = init_instance_by_config(
        cfg.model,
        feature_cols=train_dataset.feature_names,
        label_cols=[train_dataset.label_names],
        logger=logger
    )

    model.fit(train_dataset=train_dataset, val_dataset=val_dataset)
    model.save(model_dir="./temp")

    # predict stage
    model_eval = MATCCModel(logger=logger)
    model_eval.load(model_dir="./temp")
    model_eval.predict(dataset=val_dataset)
