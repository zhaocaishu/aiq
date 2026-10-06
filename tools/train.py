"""Train models with multiple random seeds."""

from __future__ import annotations

import argparse
import random
from logging import Logger
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from aiq.utils.config import config as cfg
from aiq.utils.logging import get_logger
from aiq.utils.module import init_instance_by_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cfg_file", type=Path, required=True, help="Path to the configuration file."
    )
    parser.add_argument(
        "--data_dir", type=Path, required=True, help="Directory containing data."
    )
    parser.add_argument(
        "--save_dir", type=Path, required=True, help="Directory for training artifacts."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1234],
        help="Model training seeds, e.g. 1234 2024 3407. Default: 1234.",
    )
    parser.add_argument(
        "--data_seed",
        type=int,
        default=1234,
        help="Seed for data preparation, shared across models. Default: 1234.",
    )

    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must not contain duplicates.")
    if any(not 0 <= seed < 2**32 for seed in [args.data_seed, *args.seeds]):
        parser.error("Seeds must be integers in [0, 2**32).")
    return args


def set_random_seed(seed: int) -> None:
    """Seed the global RNGs; full determinism also depends on model internals."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_datasets(
    data: Any,
    data_dir: Path,
    feature_names: Sequence[str],
) -> tuple[Any, Any]:
    train_dataset = init_instance_by_config(
        cfg.dataset,
        data=data,
        data_dir=str(data_dir),
        feature_names=feature_names,
        split="train",
    )
    val_dataset = init_instance_by_config(
        cfg.dataset,
        data=data,
        data_dir=str(data_dir),
        feature_names=feature_names,
        split="valid",
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Training and validation datasets must both be non-empty.")
    return train_dataset, val_dataset


def train_and_save_model(
    train_dataset: Any,
    val_dataset: Any,
    save_dir: Path,
    seed: int,
    logger: Logger,
) -> None:
    model = init_instance_by_config(
        cfg.model,
        feature_names=train_dataset.feature_names,
        label_names=train_dataset.label_names,
        save_dir=str(save_dir),
        logger=logger,
    )
    model.fit(train_dataset=train_dataset, val_dataset=val_dataset)

    model_name = f"model_seed_{seed}.pth"
    model.save(model_name=model_name)
    logger.info("Model saved: %s", save_dir / model_name)


def main() -> None:
    args = parse_args()
    cfg.from_file(str(args.cfg_file))

    logger = get_logger("TRAINING")
    logger.info("Configuration loaded:\n%s", cfg)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    # Fix preprocessing randomness independently of the selected model seeds.
    set_random_seed(args.data_seed)
    logger.info("Preparing data with seed %d", args.data_seed)
    data_handler = init_instance_by_config(
        cfg.data_handler,
        data_dir=str(args.data_dir),
    )
    data = data_handler.setup_data(mode="train")
    logger.info("Data prepared. Shape: %s", data.shape)

    train_dataset, val_dataset = load_datasets(
        data=data,
        data_dir=args.data_dir,
        feature_names=data_handler.feature_names,
    )
    logger.info(
        "Loaded %d training and %d validation samples.",
        len(train_dataset),
        len(val_dataset),
    )

    handler_path = args.save_dir / "data_handler.pkl"
    data_handler.save(str(handler_path))
    logger.info("Data handler saved: %s", handler_path)

    for index, seed in enumerate(args.seeds, start=1):
        logger.info("Training model %d/%d with seed %d", index, len(args.seeds), seed)
        set_random_seed(seed)
        train_and_save_model(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            save_dir=args.save_dir,
            seed=seed,
            logger=logger,
        )

    logger.info("All training runs completed successfully.")


if __name__ == "__main__":
    main()
