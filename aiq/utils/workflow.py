"""Shared training, prediction and evaluation steps with explicit configuration."""

from __future__ import annotations

import argparse
import copy
import csv
import random
from logging import Logger
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from aiq.utils.module import init_instance_by_config


def build_training_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
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
    return parser


def parse_training_args(description: str) -> argparse.Namespace:
    parser = build_training_parser(description)
    args = parser.parse_args()
    try:
        validate_seeds(args.seeds, args.data_seed)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def validate_seeds(seeds: Sequence[int], data_seed: int) -> None:
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be non-empty and unique.")
    if any(
        not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32
        for seed in [data_seed, *seeds]
    ):
        raise ValueError("Seeds must be integers in [0, 2**32).")


def set_random_seed(seed: int) -> None:
    """Seed the global RNGs; full determinism also depends on model internals."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataset(
    dataset_config: dict,
    data: Any,
    data_dir: Path,
    feature_names: Sequence[str],
    split: str,
) -> Any:
    """Build one split from already processed data."""
    if split not in dataset_config["kwargs"]["segments"]:
        raise ValueError(f"Dataset configuration has no {split!r} segment.")
    dataset = init_instance_by_config(
        dataset_config,
        data=data,
        data_dir=str(data_dir),
        feature_names=feature_names,
        split=split,
    )
    if len(dataset) == 0:
        raise ValueError(f"The {split!r} dataset must be non-empty.")
    return dataset


def save_handler(handler: Any, save_dir: Path) -> Path:
    """Save the fitted processors used by all models in this directory."""
    save_dir.mkdir(parents=True, exist_ok=True)
    handler_path = save_dir / "data_handler.pkl"
    handler.save(str(handler_path))
    if not handler_path.is_file():
        raise OSError(f"Failed to save preprocessing statistics: {handler_path}")
    return handler_path


def train_models(
    model_config: dict,
    train_dataset: Any,
    val_dataset: Any,
    save_dir: Path,
    seeds: Sequence[int],
    logger: Logger,
) -> list[str]:
    """Fit a fresh model for each seed and return its checkpoint name."""
    validate_seeds(seeds, 0)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_names = []
    for index, seed in enumerate(seeds, start=1):
        logger.info("Training model %d/%d with seed %d", index, len(seeds), seed)
        set_random_seed(seed)
        run_config = copy.deepcopy(model_config)
        if "seed" in run_config["kwargs"]:
            run_config["kwargs"]["seed"] = seed
        model = init_instance_by_config(
            run_config,
            feature_names=train_dataset.feature_names,
            label_names=train_dataset.label_names,
            save_dir=str(save_dir),
            logger=logger,
        )
        model.fit(train_dataset=train_dataset, val_dataset=val_dataset)
        model_name = f"model_seed_{seed}.pth"
        model.save(model_name=model_name)
        logger.info("Model saved: %s", save_dir / model_name)
        model_names.append(model_name)
        del model
    return model_names


def validate_model_files(model_dir: Path, model_names: Sequence[str]) -> None:
    if not model_names or len(model_names) != len(set(model_names)):
        raise ValueError("Model checkpoint names must be non-empty and unique.")
    paths = [model_dir / "data_handler.pkl"]
    paths.extend(model_dir / name for name in model_names)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Training artifact not found: {path}")


def load_model(
    model_config: dict,
    dataset: Any,
    model_dir: Path,
    model_name: str,
    logger: Logger,
) -> Any:
    logger.info("Loading model: %s", model_dir / model_name)
    model = init_instance_by_config(
        model_config,
        feature_names=dataset.feature_names,
        label_names=dataset.label_names,
        save_dir=str(model_dir),
        logger=logger,
    )
    model.load(model_name)
    return model


def prepare_predictions(
    pred_df: pd.DataFrame,
    label_names: Sequence[str],
    model_name: str,
) -> pd.DataFrame:
    """Validate predictions and index them by unique sample keys."""
    if not isinstance(pred_df, pd.DataFrame) or pred_df.empty:
        raise ValueError(f"{model_name}: predict() must return a non-empty DataFrame.")

    key_cols = ["Instrument", "Date"]
    if not all(col in pred_df.columns for col in key_cols):
        pred_df = pred_df.reset_index()
    if not pred_df.columns.is_unique:
        raise ValueError(f"{model_name}: prediction columns must be unique.")

    pred_cols = [f"PRED_{name}" for name in label_names]
    required_cols = key_cols + list(label_names) + pred_cols
    missing_cols = [col for col in required_cols if col not in pred_df.columns]
    if missing_cols:
        raise ValueError(f"{model_name}: missing prediction columns: {missing_cols}.")

    actual_pred_cols = {
        col
        for col in pred_df.columns
        if isinstance(col, str) and col.startswith("PRED_")
    }
    if actual_pred_cols != set(pred_cols):
        raise ValueError(
            f"{model_name}: prediction columns do not match dataset labels."
        )
    if pred_df[key_cols].isna().any().any():
        raise ValueError(f"{model_name}: sample keys must not contain null values.")
    if pred_df.duplicated(key_cols).any():
        raise ValueError(f"{model_name}: duplicate Instrument/Date keys.")
    if not np.isfinite(pred_df[pred_cols].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{model_name}: predictions contain NaN or infinity.")

    return pred_df.set_index(key_cols)


def load_predictions(
    pred_file: Path,
    label_names: Sequence[str],
) -> pd.DataFrame:
    """Read saved predictions while preserving stock codes and sample identity."""
    label_names = list(label_names)
    if not label_names or len(label_names) != len(set(label_names)):
        raise ValueError("Prediction label names must be non-empty and unique.")
    with pred_file.open(encoding="utf-8-sig", newline="") as stream:
        columns = next(csv.reader(stream), [])
    if not columns:
        raise ValueError(f"{pred_file}: prediction file is empty.")
    if len(columns) != len(set(columns)):
        raise ValueError(f"{pred_file}: prediction columns must be unique.")

    predictions = pd.read_csv(
        pred_file,
        dtype={"Instrument": str, "Date": str},
        float_precision="round_trip",
        encoding="utf-8-sig",
    )
    if "Date" in predictions.columns:
        predictions["Date"] = pd.to_datetime(predictions["Date"]).dt.strftime(
            "%Y-%m-%d"
        )
    for column in label_names + [f"PRED_{name}" for name in label_names]:
        if column in predictions.columns:
            predictions[column] = pd.to_numeric(predictions[column], errors="raise")
            if column in label_names and np.isinf(predictions[column]).any():
                raise ValueError(f"{pred_file}: labels must not contain infinity.")

    return prepare_predictions(predictions, label_names, str(pred_file)).reset_index()


def ensemble_predict(
    model_config: dict,
    dataset: Any,
    model_dir: Path,
    model_names: Sequence[str],
    logger: Logger,
) -> pd.DataFrame:
    """Load models one at a time and average predictions after key alignment."""
    if not model_names or len(model_names) != len(set(model_names)):
        raise ValueError("Model checkpoint names must be non-empty and unique.")
    label_names = list(dataset.label_names)
    if not label_names or len(label_names) != len(set(label_names)):
        raise ValueError("Dataset label names must be non-empty and unique.")
    pred_cols = [f"PRED_{name}" for name in label_names]
    logger.info("Ensembling prediction columns: %s", pred_cols)

    base_df = None
    pred_sum = None

    for index, model_name in enumerate(model_names, start=1):
        logger.info("Predicting with model %d/%d", index, len(model_names))
        model = load_model(model_config, dataset, model_dir, model_name, logger)
        with torch.no_grad():
            raw_predictions = model.predict(dataset)
        del model
        pred_df = prepare_predictions(raw_predictions, label_names, model_name)

        if base_df is None:
            base_df = pred_df.copy()
            pred_sum = pred_df[pred_cols].to_numpy(dtype=np.float64, copy=True)
            continue

        # Match by sample identity, never by the row numbers from reset_index().
        positions = pred_df.index.get_indexer(base_df.index)
        if len(pred_df) != len(base_df) or (positions < 0).any():
            raise ValueError(
                f"{model_name}: prediction sample keys differ across models."
            )
        pred_df = pred_df.iloc[positions]
        try:
            pd.testing.assert_frame_equal(
                base_df[label_names],
                pred_df[label_names],
                check_dtype=False,
                check_exact=True,
            )
        except AssertionError as exc:
            raise ValueError(f"{model_name}: labels differ across models.") from exc

        pred_sum += pred_df[pred_cols].to_numpy(dtype=np.float64)

    assert base_df is not None and pred_sum is not None
    base_df[pred_cols] = pred_sum / len(model_names)
    return base_df.reset_index()


def save_predictions(
    predictions: pd.DataFrame,
    label_names: Sequence[str],
    save_dir: Path,
) -> pd.DataFrame:
    """Write the common prediction schema in deterministic sample order."""
    predictions = prepare_predictions(
        predictions, label_names, "saved predictions"
    ).reset_index()
    columns = ["Instrument", "Date"] + list(label_names)
    columns.extend(f"PRED_{name}" for name in label_names)
    predictions = (
        predictions[columns].sort_values(["Date", "Instrument"]).reset_index(drop=True)
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(save_dir / "predictions.csv", index=False, float_format="%.17g")
    return predictions


def evaluate_predictions(
    evaluator_config: dict,
    predictions: pd.DataFrame,
    data_dir: Path,
    save_dir: Path,
    segment: Sequence[str],
    logger: Logger,
) -> Any:
    """Evaluate existing predictions over their selected dataset interval."""
    if len(segment) != 2:
        raise ValueError("Evaluation segment must contain a start and end date.")
    start, end = map(pd.Timestamp, segment)
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError("Evaluation dates must be valid and increasing.")
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        raise ValueError("Evaluation requires non-empty predictions.")
    if "Date" not in predictions.columns:
        raise ValueError("Evaluation predictions must include Date.")
    dates = pd.to_datetime(predictions["Date"])
    if dates.isna().any() or not dates.between(start, end).all():
        raise ValueError("Prediction dates fall outside the evaluation segment.")

    evaluation_config = copy.deepcopy(evaluator_config)
    evaluation_config["kwargs"].update(
        start_time=start.strftime("%Y-%m-%d"),
        end_time=end.strftime("%Y-%m-%d"),
    )
    evaluator = init_instance_by_config(
        evaluation_config, data_dir=str(data_dir), logger=logger
    )
    metrics = evaluator.evaluate(predictions)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "metrics.txt").write_text(str(metrics), encoding="utf-8")
    logger.info("Evaluation metrics:\n%s", metrics)
    return metrics
