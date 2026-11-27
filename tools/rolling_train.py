"""Train monthly rolling models without validation and predict the test period."""

from __future__ import annotations

import copy
import sys
from logging import Logger
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yaml

from aiq.dataset.loader import DataLoader
from aiq.utils.logging import get_logger
from aiq.utils.module import init_instance_by_config
from aiq.utils import workflow


TRAIN_MONTHS = 72
LABEL_HORIZON = 5


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
    test_start, test_end = map(pd.Timestamp, segments["test"])
    if test_start > test_end:
        raise ValueError("Test date range must be increasing.")
    labels = config["dataset"]["kwargs"]["label_names"]
    if labels != ["RET_5D"]:
        raise ValueError("Monthly rolling training requires RET_5D as the only label.")

    first_train_start = (test_start.to_period("M") - TRAIN_MONTHS).start_time
    warmup_history = first_train_start - pd.Timestamp(
        config["data_handler"]["kwargs"]["start_time"]
    )
    if warmup_history < pd.Timedelta(0):
        raise ValueError("Handler history must start before the training window.")

    prediction_start = max(test_start, month.start_time.normalize())
    prediction_end = min(test_end, month.end_time.normalize())
    first_prediction_index = calendar.searchsorted(prediction_start, side="left")
    last_prediction_index = calendar.searchsorted(prediction_end, side="right") - 1
    label_end_index = last_prediction_index + LABEL_HORIZON
    cutoff_index = calendar.searchsorted(month.start_time, side="left") - 1
    if (
        prediction_start > prediction_end
        or first_prediction_index > last_prediction_index
    ):
        raise ValueError(f"{month}: prediction month contains no trading days.")
    if cutoff_index < LABEL_HORIZON or label_end_index >= len(calendar):
        raise ValueError(
            f"{month}: calendar does not cover the five-day label horizon."
        )

    # A signal at t uses prices at t+1 and t+5; both must be known at cutoff.
    train_start = (month - TRAIN_MONTHS).start_time
    train_end = calendar[cutoff_index - LABEL_HORIZON]
    if train_start > train_end or train_start < calendar[0]:
        raise ValueError(f"{month}: calendar does not cover the training window.")

    train_config = copy.deepcopy(config)
    handler_kwargs = train_config["data_handler"]["kwargs"]
    handler_kwargs["start_time"] = (train_start - warmup_history).strftime("%Y-%m-%d")
    handler_kwargs["end_time"] = calendar[cutoff_index].strftime("%Y-%m-%d")
    handler_kwargs["fit_start_time"] = train_start.strftime("%Y-%m-%d")
    handler_kwargs["fit_end_time"] = train_end.strftime("%Y-%m-%d")
    train_config["dataset"]["kwargs"]["segments"] = {
        "train": [handler_kwargs["fit_start_time"], handler_kwargs["fit_end_time"]],
        "test": [
            prediction_start.strftime("%Y-%m-%d"),
            prediction_end.strftime("%Y-%m-%d"),
        ],
    }
    predict_config = copy.deepcopy(train_config)
    # Extra quotes are used only for offline labels, never for fitting statistics.
    predict_config["data_handler"]["kwargs"]["end_time"] = calendar[
        label_end_index
    ].strftime("%Y-%m-%d")
    return train_config, predict_config


def slice_raw_data(
    raw_data: dict,
    calendar: Sequence[str],
    segment: Sequence[str],
    seq_len: int,
) -> dict:
    """Keep sample dates and sequence context without discarding factor history."""
    start, end = segment
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar)).unique().sort_values()
    first_sample = calendar.searchsorted(pd.Timestamp(start))
    context_start = calendar[max(0, first_sample - seq_len + 1)].strftime("%Y-%m-%d")
    data = dict(raw_data)
    for name in ("instrument", "market"):
        if name in data:
            data[name] = data[name].loc[context_start:end]
    return data


def prepare_dataset(
    handler: Any,
    raw_data: dict,
    dataset_config: dict,
    data_dir: Path,
    split: str,
) -> Any:
    """Apply window preprocessing and construct the requested dataset."""
    dataset_kwargs = dataset_config["kwargs"]
    window_data = slice_raw_data(
        raw_data,
        handler.calendar,
        dataset_kwargs["segments"][split],
        dataset_kwargs["seq_len"],
    )
    data = handler.process_data(window_data, mode=split)
    return workflow.create_dataset(
        dataset_config, data, data_dir, handler.feature_names, split=split
    )


def prepare_raw_data(
    handler_config: dict,
    month_configs: Sequence[tuple[pd.Period, dict, dict]],
    data_dir: Path,
    data_seed: int,
    logger: Logger,
) -> dict:
    """Extract one raw cache covering every monthly window and offline label."""
    raw_config = copy.deepcopy(handler_config)
    handler_kwargs = raw_config["kwargs"]
    handler_kwargs["start_time"] = min(
        train["data_handler"]["kwargs"]["start_time"] for _, train, _ in month_configs
    )
    handler_kwargs["end_time"] = max(
        predict["data_handler"]["kwargs"]["end_time"] for _, _, predict in month_configs
    )
    workflow.set_random_seed(data_seed)
    handler = init_instance_by_config(raw_config, data_dir=str(data_dir))
    logger.info(
        "Extracting raw factors once for %s to %s.",
        handler_kwargs["start_time"],
        handler_kwargs["end_time"],
    )
    return handler.fetch_data()


def run_month(
    train_config: dict,
    predict_config: dict,
    data_dir: Path,
    save_dir: Path,
    seeds: Sequence[int],
    data_seed: int,
    logger: Logger,
    raw_data: dict,
) -> pd.DataFrame:
    """Train fresh models, then predict one month with their fitted preprocessors."""
    workflow.validate_seeds(seeds, data_seed)
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, config in [
        ("train_config.yaml", train_config),
        ("predict_config.yaml", predict_config),
    ]:
        (save_dir / name).write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    workflow.set_random_seed(data_seed)
    handler = init_instance_by_config(
        train_config["data_handler"], data_dir=str(data_dir)
    )
    train_dataset = prepare_dataset(
        handler, raw_data, train_config["dataset"], data_dir, split="train"
    )
    logger.info(
        "Training on %d daily cross-sections without validation.", len(train_dataset)
    )
    handler_path = workflow.save_handler(handler, save_dir)
    model_names = workflow.train_models(
        train_config["model"], train_dataset, None, save_dir, seeds, logger
    )
    del train_dataset, handler

    workflow.validate_model_files(save_dir, model_names)
    workflow.set_random_seed(data_seed)
    handler = init_instance_by_config(
        predict_config["data_handler"], data_dir=str(data_dir)
    )
    handler.load(str(handler_path))
    test_dataset = prepare_dataset(
        handler, raw_data, predict_config["dataset"], data_dir, split="test"
    )
    predictions = workflow.ensemble_predict(
        predict_config["model"], test_dataset, save_dir, model_names, logger
    )
    return workflow.save_predictions(predictions, test_dataset.label_names, save_dir)


def run_rolling(
    config: dict,
    data_dir: Path,
    save_dir: Path,
    seeds: Sequence[int],
    data_seed: int,
    logger: Logger,
) -> pd.DataFrame:
    """Train monthly models and save predictions for the complete test period."""
    workflow.validate_seeds(seeds, data_seed)
    start, end = config["dataset"]["kwargs"]["segments"]["test"]
    handler_kwargs = config["data_handler"]["kwargs"]
    calendar = sorted(
        set(
            DataLoader.load_calendar(
                str(data_dir),
                start_time=handler_kwargs["start_time"],
                end_time=handler_kwargs["end_time"],
            )
        )
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    month_configs = [
        (month, *build_month_configs(config, calendar, month))
        for month in pd.period_range(start=start, end=end, freq="M")
    ]

    raw_data = prepare_raw_data(
        config["data_handler"], month_configs, data_dir, data_seed, logger
    )

    monthly_predictions = []
    for month, train_config, predict_config in month_configs:
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
                raw_data,
            )
        )

    del raw_data

    labels = config["dataset"]["kwargs"]["label_names"]
    predictions = workflow.prepare_predictions(
        pd.concat(monthly_predictions, ignore_index=True),
        labels,
        "rolling predictions",
    ).reset_index()
    expected_dates = {date for date in calendar if start <= date <= end}
    if set(predictions["Date"]) != expected_dates:
        raise ValueError("Rolling predictions do not cover every test trading day.")
    predictions = workflow.save_predictions(predictions, labels, save_dir)
    logger.info("Rolling predictions saved: %s", save_dir / "predictions.csv")
    return predictions


def main() -> None:
    args = workflow.parse_training_args(description=__doc__)
    config: dict[str, Any] = yaml.safe_load(args.cfg_file.read_text(encoding="utf-8"))
    logger = get_logger("ROLLING_TRAINING")
    run_rolling(
        config, args.data_dir, args.save_dir, args.seeds, args.data_seed, logger
    )


if __name__ == "__main__":
    main()
