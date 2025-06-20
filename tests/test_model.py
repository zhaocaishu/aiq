from aiq.utils.config import config as cfg
from aiq.utils.module import init_instance_by_config
from aiq.utils.logging import get_logger


if __name__ == "__main__":
    # config
    cfg.from_file("./configs/ppnet_model_reg.yaml")

    # logger
    logger = get_logger("TEST_MODEL")

    # data handler
    data_handler = init_instance_by_config(cfg.data_handler, data_dir="./data")
    data = data_handler.setup_data()

    # train dataset
    train_dataset = init_instance_by_config(
        cfg.dataset,
        data=data,
        feature_names=data_handler.feature_names,
        mode="train",
    )

    val_dataset = init_instance_by_config(
        cfg.dataset,
        data=data,
        feature_names=data_handler.feature_names,
        mode="valid",
    )

    print(train_dataset.feature_names, train_dataset.label_names)

    # train stage
    model = init_instance_by_config(
        cfg.model,
        feature_names=train_dataset.feature_names,
        label_names=train_dataset.label_names,
        save_dir="./checkpoints",
        logger=logger,
    )

    model.fit(train_dataset=train_dataset, val_dataset=val_dataset)
    model.save()

    # predict stage
    model.load()
    pred_df = model.predict(test_dataset=val_dataset)
    print(pred_df.head(3))
