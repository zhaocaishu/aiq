"""Train models with multiple random seeds."""

from __future__ import annotations

import sys
from logging import Logger
from pathlib import Path
from typing import Sequence

import yaml

from aiq.utils import workflow
from aiq.utils.logging import get_logger
from aiq.utils.module import init_instance_by_config


def run_training(
    config: dict,
    data_dir: Path,
    save_dir: Path,
    seeds: Sequence[int],
    data_seed: int,
    logger: Logger,
) -> list[str]:
    """Prepare training/validation data once and fit independent seeded models."""
    workflow.validate_seeds(seeds, data_seed)
    segments = config["dataset"]["kwargs"]["segments"]
    if not all(split in segments for split in ("train", "valid")):
        raise ValueError("Ordinary training requires train and valid segments.")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Fix preprocessing randomness independently of the selected model seeds.
    workflow.set_random_seed(data_seed)
    logger.info("Preparing data with seed %d", data_seed)
    handler = init_instance_by_config(config["data_handler"], data_dir=str(data_dir))
    data = handler.setup_data(mode="train")
    logger.info("Data prepared. Shape: %s", data.shape)
    train_dataset = workflow.create_dataset(
        config["dataset"], data, data_dir, handler.feature_names, split="train"
    )
    val_dataset = workflow.create_dataset(
        config["dataset"], data, data_dir, handler.feature_names, split="valid"
    )
    logger.info(
        "Loaded %d training and %d validation samples.",
        len(train_dataset),
        len(val_dataset),
    )
    handler_path = workflow.save_handler(handler, save_dir)
    logger.info("Data handler saved: %s", handler_path)
    model_names = workflow.train_models(
        config["model"], train_dataset, val_dataset, save_dir, seeds, logger
    )
    logger.info("All training runs completed successfully.")
    return model_names


def main() -> None:
    args = workflow.parse_training_args(description=__doc__)
    config = yaml.safe_load(args.cfg_file.read_text(encoding="utf-8"))
    logger = get_logger("TRAINING")
    logger.info("Configuration loaded:\n%s", config)
    run_training(
        config, args.data_dir, args.save_dir, args.seeds, args.data_seed, logger
    )


if __name__ == "__main__":
    main()
