"""Predict with saved models and evaluate the results or an existing prediction CSV."""

from __future__ import annotations

import argparse
import sys
from logging import Logger
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yaml

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aiq.utils import workflow
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
        "--save_dir",
        type=Path,
        required=True,
        help="Directory for predictions and evaluation outputs.",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        help="Directory containing checkpoints and data_handler.pkl. Defaults to save_dir.",
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        help="Checkpoints to average. Default: model_seed_1234.pth.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "valid", "test"),
        default="test",
        help="Dataset split to predict and evaluate. Default: test.",
    )
    parser.add_argument(
        "--data_seed",
        type=int,
        default=1234,
        help="Seed for prediction data preparation. Default: 1234.",
    )
    parser.add_argument(
        "--pred_file",
        type=Path,
        help="Evaluate an existing CSV with raw return labels instead of running models.",
    )
    args = parser.parse_args()
    if args.pred_file is not None:
        if args.model_dir is not None or args.model_names is not None:
            parser.error(
                "--pred_file cannot be combined with --model_dir or --model_names."
            )
    else:
        try:
            workflow.validate_seeds([args.data_seed], args.data_seed)
        except ValueError as exc:
            parser.error(str(exc))
        if args.model_dir is None:
            args.model_dir = args.save_dir
        if args.model_names is None:
            args.model_names = ["model_seed_1234.pth"]
        if len(args.model_names) != len(set(args.model_names)):
            parser.error("--model_names must not contain duplicates.")
    return args


def run_prediction(
    config: dict,
    data_dir: Path,
    model_dir: Path,
    model_names: Sequence[str],
    split: str,
    data_seed: int,
    logger: Logger,
) -> pd.DataFrame:
    """Predict a split using saved preprocessing statistics and raw labels."""
    if split not in config["dataset"]["kwargs"]["segments"]:
        raise ValueError(f"Dataset configuration has no {split!r} segment.")
    workflow.validate_seeds([data_seed], data_seed)
    workflow.validate_model_files(model_dir, model_names)
    workflow.set_random_seed(data_seed)
    logger.info("Preparing data with seed %d", data_seed)
    handler = init_instance_by_config(config["data_handler"], data_dir=str(data_dir))
    handler.load(str(model_dir / "data_handler.pkl"))
    # Evaluation of train/valid also uses inference preprocessing and raw labels.
    data = handler.setup_data(mode="test")
    dataset = workflow.create_dataset(
        config["dataset"], data, data_dir, handler.feature_names, split=split
    )
    logger.info("Loaded %d %s samples.", len(dataset), split)
    return workflow.ensemble_predict(
        config["model"], dataset, model_dir, model_names, logger
    )


def run_evaluation(
    config: dict,
    data_dir: Path,
    save_dir: Path,
    pred_file: Path,
    split: str,
    logger: Logger,
) -> Any:
    """Read saved raw-label predictions and evaluate the requested interval."""
    dataset_kwargs = config["dataset"]["kwargs"]
    if split not in dataset_kwargs["segments"]:
        raise ValueError(f"Dataset configuration has no {split!r} segment.")
    predictions = workflow.load_predictions(pred_file, dataset_kwargs["label_names"])
    return workflow.evaluate_predictions(
        config["evaluator"],
        predictions,
        data_dir,
        save_dir,
        dataset_kwargs["segments"][split],
        logger,
    )


def run_test(
    config: dict,
    data_dir: Path,
    save_dir: Path,
    model_dir: Path,
    model_names: Sequence[str],
    split: str,
    data_seed: int,
    logger: Logger,
) -> Any:
    """Save model predictions, then evaluate the same file used for later reviews."""
    predictions = run_prediction(
        config, data_dir, model_dir, model_names, split, data_seed, logger
    )
    workflow.save_predictions(
        predictions, config["dataset"]["kwargs"]["label_names"], save_dir
    )
    return run_evaluation(
        config, data_dir, save_dir, save_dir / "predictions.csv", split, logger
    )


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.cfg_file.read_text(encoding="utf-8"))
    logger = get_logger("TESTING")
    logger.info("Configuration loaded:\n%s", config)
    if args.pred_file is not None:
        run_evaluation(
            config, args.data_dir, args.save_dir, args.pred_file, args.split, logger
        )
    else:
        run_test(
            config,
            args.data_dir,
            args.save_dir,
            args.model_dir,
            args.model_names,
            args.split,
            args.data_seed,
            logger,
        )


if __name__ == "__main__":
    main()
