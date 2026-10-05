"""Evaluate one model or an ensemble of model checkpoints."""

from __future__ import annotations

import argparse
import random
from logging import Logger
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
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
        "--save_dir", type=Path, required=True, help="Directory for evaluation outputs."
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        help=(
            "Directory containing checkpoints and data_handler.pkl. "
            "Defaults to save_dir."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("train", "valid", "test"),
        default="test",
        help="Dataset split to evaluate. Default: test.",
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        default=["model_seed_1234.pth"],
        help="Checkpoint names to average. Default: model_seed_1234.pth.",
    )
    parser.add_argument(
        "--data_seed",
        type=int,
        default=1234,
        help="Seed for data preparation. Default: 1234.",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save keys, labels and ensemble scores to save_dir/predictions.csv.",
    )

    args = parser.parse_args()
    if len(args.model_names) != len(set(args.model_names)):
        parser.error("--model_names must not contain duplicates.")
    if not 0 <= args.data_seed < 2**32:
        parser.error("--data_seed must be an integer in [0, 2**32).")
    if args.model_dir is None:
        args.model_dir = args.save_dir
    return args


def set_random_seed(seed: int) -> None:
    """Seed the global RNGs; full determinism also depends on model internals."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(
    dataset: Any,
    model_dir: Path,
    model_name: str,
    logger: Logger,
) -> Any:
    logger.info("Loading model: %s", model_dir / model_name)
    model = init_instance_by_config(
        cfg.model,
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


def ensemble_predict(
    dataset: Any,
    model_dir: Path,
    model_names: Sequence[str],
    logger: Logger,
) -> pd.DataFrame:
    """Load models one at a time and average predictions after key alignment."""
    if not model_names:
        raise ValueError("At least one model checkpoint is required.")
    label_names = list(dataset.label_names)
    if not label_names or len(label_names) != len(set(label_names)):
        raise ValueError("Dataset label names must be non-empty and unique.")
    pred_cols = [f"PRED_{name}" for name in label_names]
    logger.info("Ensembling prediction columns: %s", pred_cols)

    base_df = None
    pred_sum = None

    for index, model_name in enumerate(model_names, start=1):
        logger.info("Predicting with model %d/%d", index, len(model_names))
        model = load_model(dataset, model_dir, model_name, logger)
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


def main() -> None:
    args = parse_args()
    cfg.from_file(str(args.cfg_file))

    logger = get_logger("EVALUATION")
    logger.info("Configuration loaded:\n%s", cfg)

    handler_path = args.model_dir / "data_handler.pkl"
    for path in [handler_path, *(args.model_dir / name for name in args.model_names)]:
        if not path.is_file():
            raise FileNotFoundError(f"Training artifact not found: {path}")
    args.save_dir.mkdir(parents=True, exist_ok=True)

    set_random_seed(args.data_seed)
    logger.info("Preparing data with seed %d", args.data_seed)
    data_handler = init_instance_by_config(
        cfg.data_handler,
        data_dir=str(args.data_dir),
    )
    data_handler.load(str(handler_path))
    eval_data = data_handler.setup_data(mode="test")
    logger.info("Data prepared. Shape: %s", eval_data.shape)

    eval_dataset = init_instance_by_config(
        cfg.dataset,
        data=eval_data,
        data_dir=str(args.data_dir),
        feature_names=data_handler.feature_names,
        split=args.split,
    )
    if len(eval_dataset) == 0:
        raise ValueError(f"The {args.split!r} dataset must be non-empty.")
    logger.info("Loaded %d %s samples.", len(eval_dataset), args.split)

    pred_df = ensemble_predict(
        dataset=eval_dataset,
        model_dir=args.model_dir,
        model_names=args.model_names,
        logger=logger,
    )
    logger.info("Ensemble prediction completed. Shape: %s", pred_df.shape)

    if args.save_predictions:
        label_names = list(eval_dataset.label_names)
        output_cols = (
            ["Instrument", "Date"]
            + label_names
            + [f"PRED_{name}" for name in label_names]
        )
        prediction_path = args.save_dir / "predictions.csv"
        pred_df[output_cols].to_csv(prediction_path, index=False)
        logger.info("Predictions saved: %s", prediction_path)

    evaluator = init_instance_by_config(
        cfg.evaluator,
        data_dir=str(args.data_dir),
        logger=logger,
    )
    metrics = evaluator.evaluate(pred_df)
    logger.info("Evaluation metrics:\n%s", metrics)


if __name__ == "__main__":
    main()
