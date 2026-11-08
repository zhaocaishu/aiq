"""Train monthly rolling models without validation and predict the test period."""

from __future__ import annotations

import copy
import sys
from logging import Logger
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yaml

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aiq.dataset.loader import DataLoader
from aiq.utils.config import Config, config as cfg
from aiq.utils.logging import get_logger
from aiq.utils.module import init_instance_by_config

from tools.eval import ensemble_predict, prepare_predictions
from tools.train import parse_args, set_random_seed, train_and_save_model


def build_month_configs(
    config: dict,
    calendar: Sequence[str],
    month: pd.Period,
) -> tuple[dict, dict]:
    """Shift a fixed training window and purge labels at the monthly cutoff."""
    calendar = (
        pd.DatetimeIndex(pd.to_datetime(calendar)).normalize().unique().sort_values()
    )
    month = pd.Period(month, freq="M")
    segments = config["dataset"]["kwargs"]["segments"]
    train_start, train_end = map(pd.Timestamp, segments["train"])
    test_start, test_end = map(pd.Timestamp, segments["test"])
    if train_start > train_end or test_start > test_end:
        raise ValueError("Training and test date ranges must be increasing.")
    if config["dataset"]["kwargs"]["label_names"] != ["RET_5D"]:
        raise ValueError("Monthly rolling training requires the RET_5D label.")

    train_months = (train_end.year - train_start.year) * 12
    train_months += train_end.month - train_start.month + 1
    history = train_start - pd.Timestamp(config["data_handler"]["kwargs"]["start_time"])
    if history < pd.Timedelta(0):
        raise ValueError("Handler history must start before the training window.")

    prediction_start = max(test_start, month.start_time.normalize())
    prediction_end = min(test_end, month.end_time.normalize())
    first_prediction = calendar.searchsorted(prediction_start, side="left")
    last_prediction = calendar.searchsorted(prediction_end, side="right") - 1
    cutoff = calendar.searchsorted(month.start_time, side="left") - 1
    if prediction_start > prediction_end or first_prediction > last_prediction:
        raise ValueError(f"{month}: prediction month contains no trading days.")
    if cutoff < 5 or last_prediction + 5 >= len(calendar):
        raise ValueError(
            f"{month}: calendar does not cover the five-day label horizon."
        )

    # A signal at t uses prices at t+1 and t+5; both must be known at cutoff.
    rolling_start = month.start_time.normalize() - pd.DateOffset(months=train_months)
    rolling_end = calendar[cutoff - 5]
    if rolling_start > rolling_end or rolling_start < calendar[0]:
        raise ValueError(f"{month}: calendar does not cover the training window.")

    train_config = copy.deepcopy(config)
    handler = train_config["data_handler"]["kwargs"]
    handler["start_time"] = (rolling_start - history).strftime("%Y-%m-%d")
    handler["end_time"] = calendar[cutoff].strftime("%Y-%m-%d")
    handler["fit_start_time"] = rolling_start.strftime("%Y-%m-%d")
    handler["fit_end_time"] = rolling_end.strftime("%Y-%m-%d")
    train_config["dataset"]["kwargs"]["segments"] = {
        "train": [handler["fit_start_time"], handler["fit_end_time"]],
        "test": [
            prediction_start.strftime("%Y-%m-%d"),
            prediction_end.strftime("%Y-%m-%d"),
        ],
    }
    train_config["evaluator"]["kwargs"].update(
        start_time=prediction_start.strftime("%Y-%m-%d"),
        end_time=prediction_end.strftime("%Y-%m-%d"),
    )

    predict_config = copy.deepcopy(train_config)
    # Extra quotes are used only for offline labels, never for fitting statistics.
    predict_config["data_handler"]["kwargs"]["end_time"] = calendar[
        last_prediction + 5
    ].strftime("%Y-%m-%d")
    return train_config, predict_config


def run_month(
    train_config: dict,
    predict_config: dict,
    data_dir: Path,
    save_dir: Path,
    seeds: Sequence[int],
    data_seed: int,
    logger: Logger,
) -> pd.DataFrame:
    """Train fresh models, then predict one month with their fitted preprocessors."""
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, config in [
        ("train_config.yaml", train_config),
        ("predict_config.yaml", predict_config),
    ]:
        (save_dir / name).write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    cfg.from_config(Config(train_config))
    set_random_seed(data_seed)
    handler = init_instance_by_config(cfg.data_handler, data_dir=str(data_dir))
    data = handler.setup_data(mode="train")
    train_dataset = init_instance_by_config(
        cfg.dataset,
        data=data,
        data_dir=str(data_dir),
        feature_names=handler.feature_names,
        split="train",
    )
    if len(train_dataset) == 0:
        raise ValueError(f"{save_dir.name}: training dataset is empty.")
    logger.info(
        "Training on %d daily cross-sections without validation.", len(train_dataset)
    )
    handler_path = save_dir / "data_handler.pkl"
    handler.save(str(handler_path))
    if not handler_path.is_file():
        raise OSError(f"Failed to save preprocessing statistics: {handler_path}")

    for index, seed in enumerate(seeds, start=1):
        logger.info("Training seed %d (%d/%d)", seed, index, len(seeds))
        set_random_seed(seed)
        train_and_save_model(train_dataset, None, save_dir, seed, logger)
    del train_dataset, data, handler

    cfg.from_config(Config(predict_config))
    set_random_seed(data_seed)
    handler = init_instance_by_config(cfg.data_handler, data_dir=str(data_dir))
    handler.load(str(handler_path))
    data = handler.setup_data(mode="test")
    test_dataset = init_instance_by_config(
        cfg.dataset,
        data=data,
        data_dir=str(data_dir),
        feature_names=handler.feature_names,
        split="test",
    )
    if len(test_dataset) == 0:
        raise ValueError(f"{save_dir.name}: prediction dataset is empty.")
    predictions = ensemble_predict(
        test_dataset, save_dir, [f"model_seed_{seed}.pth" for seed in seeds], logger
    )
    labels = list(test_dataset.label_names)
    columns = ["Instrument", "Date"] + labels + [f"PRED_{name}" for name in labels]
    predictions = predictions[columns].sort_values(["Date", "Instrument"])
    predictions.to_csv(save_dir / "predictions.csv", index=False)
    return predictions


def run_rolling(
    config: dict,
    data_dir: Path,
    save_dir: Path,
    seeds: Sequence[int],
    data_seed: int,
    logger: Logger,
) -> pd.DataFrame:
    """Combine monthly predictions and evaluate one continuous test period."""
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be non-empty and unique.")
    start, end = config["dataset"]["kwargs"]["segments"]["test"]
    calendar_end = max(
        pd.Timestamp(config["data_handler"]["kwargs"]["end_time"]),
        pd.Timestamp(end) + pd.DateOffset(months=1),
    ).strftime("%Y-%m-%d")
    calendar = sorted(
        set(
            DataLoader.load_calendar(
                str(data_dir),
                start_time=config["data_handler"]["kwargs"]["start_time"],
                end_time=calendar_end,
            )
        )
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    monthly_predictions = []
    for month in pd.period_range(start=start, end=end, freq="M"):
        train_config, predict_config = build_month_configs(config, calendar, month)
        segments = train_config["dataset"]["kwargs"]["segments"]
        logger.info(
            "Month %s | train %s | predict %s",
            month,
            segments["train"],
            segments["test"],
        )
        monthly_predictions.append(
            run_month(
                train_config,
                predict_config,
                data_dir,
                save_dir / str(month),
                seeds,
                data_seed,
                logger,
            )
        )

    labels = config["dataset"]["kwargs"]["label_names"]
    predictions = (
        prepare_predictions(
            pd.concat(monthly_predictions, ignore_index=True),
            labels,
            "rolling predictions",
        )
        .reset_index()
        .sort_values(["Date", "Instrument"])
        .reset_index(drop=True)
    )
    expected_dates = {date for date in calendar if start <= date <= end}
    if set(predictions["Date"]) != expected_dates:
        raise ValueError("Rolling predictions do not cover every test trading day.")
    predictions.to_csv(save_dir / "predictions.csv", index=False)

    evaluation_config = copy.deepcopy(config)
    evaluation_config["evaluator"]["kwargs"].update(start_time=start, end_time=end)
    cfg.from_config(Config(evaluation_config))
    evaluator = init_instance_by_config(
        cfg.evaluator, data_dir=str(data_dir), logger=logger
    )
    metrics = evaluator.evaluate(predictions)
    (save_dir / "metrics.txt").write_text(str(metrics), encoding="utf-8")
    logger.info("Rolling evaluation metrics:\n%s", metrics)
    return predictions


def main() -> None:
    args = parse_args(description=__doc__)
    config: dict[str, Any] = yaml.safe_load(args.cfg_file.read_text(encoding="utf-8"))
    logger = get_logger("ROLLING_TRAINING")
    run_rolling(
        config, args.data_dir, args.save_dir, args.seeds, args.data_seed, logger
    )


if __name__ == "__main__":
    main()
