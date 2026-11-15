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
    base_train_start, base_train_end = map(pd.Timestamp, segments["train"])
    test_start, test_end = map(pd.Timestamp, segments["test"])
    if base_train_start > base_train_end or test_start > test_end:
        raise ValueError("Training and test date ranges must be increasing.")
    if config["dataset"]["kwargs"]["label_names"] != ["RET_5D"]:
        raise ValueError("Monthly rolling training requires the RET_5D label.")

    train_months = (base_train_end.year - base_train_start.year) * 12
    train_months += base_train_end.month - base_train_start.month + 1
    warmup_history = base_train_start - pd.Timestamp(
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
    train_start = month.start_time.normalize() - pd.DateOffset(months=train_months)
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
    train_config["evaluator"]["kwargs"].update(
        start_time=prediction_start.strftime("%Y-%m-%d"),
        end_time=prediction_end.strftime("%Y-%m-%d"),
    )

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
    return init_instance_by_config(
        dataset_config,
        data=data,
        data_dir=str(data_dir),
        feature_names=handler.feature_names,
        split=split,
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
    set_random_seed(data_seed)
    handler = init_instance_by_config(Config(raw_config), data_dir=str(data_dir))
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
    train_dataset = prepare_dataset(
        handler, raw_data, cfg.dataset, data_dir, split="train"
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
    del train_dataset, handler

    cfg.from_config(Config(predict_config))
    set_random_seed(data_seed)
    handler = init_instance_by_config(cfg.data_handler, data_dir=str(data_dir))
    handler.load(str(handler_path))
    test_dataset = prepare_dataset(
        handler, raw_data, cfg.dataset, data_dir, split="test"
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
