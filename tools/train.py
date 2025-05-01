import os
import argparse
import pickle
from typing import Any

from aiq.utils.config import config as cfg
from aiq.utils.module import init_instance_by_config
from aiq.utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--cfg_file", type=str, required=True, help="Path to training config file"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory containing training data"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save model and data handler",
    )
    return parser.parse_args()


def setup_logger(name: str = "TRAINING") -> Any:
    return get_logger(name)


def setup_config(cfg_file: str) -> None:
    cfg.from_file(cfg_file)


def setup_directories(save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)


def load_datasets(data_dir: str, data_handler: Any) -> tuple:
    train_dataset = init_instance_by_config(
        cfg.dataset, data_dir=data_dir, data_handler=data_handler, mode="train"
    )
    val_dataset = init_instance_by_config(
        cfg.dataset, data_dir=data_dir, data_handler=data_handler, mode="valid"
    )
    return train_dataset, val_dataset


def save_data_handler(handler: Any, save_path: str) -> None:
    with open(save_path, "wb") as f:
        pickle.dump(handler, f)


def train_and_save_model(
    train_dataset: Any, val_dataset: Any, save_dir: str, logger: Any
) -> None:
    model = init_instance_by_config(
        cfg.model,
        feature_cols=train_dataset.feature_names,
        label_cols=train_dataset.label_names,
        save_dir=save_dir,
        logger=logger,
    )
    model.fit(train_dataset=train_dataset, val_dataset=val_dataset)
    model.save()


def main():
    args = parse_args()
    logger = setup_logger()

    try:
        setup_config(args.cfg_file)
        logger.info("Configuration loaded:\n%s", cfg)

        setup_directories(args.save_dir)

        data_handler = init_instance_by_config(cfg.data_handler)
        train_dataset, val_dataset = load_datasets(args.data_dir, data_handler)

        save_data_handler(data_handler, os.path.join(args.save_dir, "data_handler.pkl"))

        logger.info(
            "Loaded %d training and %d validation samples.",
            len(train_dataset),
            len(val_dataset),
        )

        train_and_save_model(train_dataset, val_dataset, args.save_dir, logger)

        logger.info("Model training completed successfully!")

    except Exception as e:
        logger.exception("Training failed due to an error: %s", str(e))


if __name__ == "__main__":
    main()
