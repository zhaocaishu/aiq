import os
import argparse
import random
from typing import List, Any

import torch
import numpy as np

from aiq.utils.config import config as cfg
from aiq.utils.module import init_instance_by_config
from aiq.utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train model with multiple seeds")
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
        help="Directory to save models and data handler",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1234],
        help="Random seeds for training (support multiple seeds), e.g. 1234 2024 3407",
    )
    return parser.parse_args()


def setup_logger(name: str = "TRAINING") -> Any:
    return get_logger(name)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_datasets(data: Any, data_dir: str, feature_names: List[str]) -> tuple:
    train_dataset = init_instance_by_config(
        cfg.dataset,
        data=data,
        data_dir=data_dir,
        feature_names=feature_names,
        mode="train",
    )

    val_dataset = init_instance_by_config(
        cfg.dataset,
        data=data,
        data_dir=data_dir,
        feature_names=feature_names,
        mode="valid",
    )
    return train_dataset, val_dataset


def train_and_save_model(
    train_dataset: Any,
    val_dataset: Any,
    save_dir: str,
    logger: Any,
    seed: int,
) -> None:
    logger.info(f"Start training with seed={seed}")

    model = init_instance_by_config(
        cfg.model,
        feature_names=train_dataset.feature_names,
        label_names=train_dataset.label_names,
        save_dir=save_dir,
        logger=logger,
    )

    model.fit(train_dataset=train_dataset, val_dataset=val_dataset)

    model_name = f"model_seed_{seed}.pth"
    model.save(model_name=model_name)


def main():
    args = parse_args()

    # Load config
    cfg.from_file(args.cfg_file)

    logger = setup_logger("TRAINING")
    logger.info("Configuration loaded:\n%s", cfg)

    os.makedirs(args.save_dir, exist_ok=True)

    # Data preparation
    data_handler = init_instance_by_config(cfg.data_handler, data_dir=args.data_dir)
    data = data_handler.setup_data()
    logger.info("Data handler completed. Shape: %s", data.shape)

    data_handler.save(os.path.join(args.save_dir, "data_handler.pkl"))

    train_dataset, val_dataset = load_datasets(
        data, args.data_dir, data_handler.feature_names
    )

    logger.info(
        "Loaded %d training and %d validation samples.",
        len(train_dataset),
        len(val_dataset),
    )

    # Multiple seeds training
    for seed in args.seeds:
        logger.info("-" * 30)
        logger.info(f"Starting training with SEED: {seed}")
        logger.info("-" * 30)

        set_random_seed(seed)
        train_and_save_model(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            save_dir=args.save_dir,
            logger=logger,
            seed=seed,
        )

    logger.info("All seed trainings completed successfully!")


if __name__ == "__main__":
    main()
