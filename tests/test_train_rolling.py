import copy
import logging
from pathlib import Path

import pandas as pd
import pytest
import yaml

from tools import train_rolling


@pytest.fixture
def config():
    path = Path(__file__).resolve().parents[1] / "configs" / "ppnet_model_reg.yaml"
    with path.open() as stream:
        return yaml.safe_load(stream)


@pytest.fixture
def calendar():
    path = Path(__file__).resolve().parents[1] / "data" / "calendars" / "trade_days.csv"
    dates = pd.read_csv(path)
    return sorted(dates.loc[dates["Exchange"] == "SSE", "Date"].unique().tolist())


def test_monthly_windows_use_trading_day_label_gap(config, calendar):
    original = copy.deepcopy(config)
    expected = {
        "2023-01": ("2017-01-01", "2022-12-23", "2022-12-30", "2023-02-07"),
        "2023-02": ("2017-02-01", "2023-01-17", "2023-01-31", "2023-03-07"),
        "2023-12": ("2017-12-01", "2023-11-23", "2023-11-30", "2024-01-08"),
    }

    for month, (start, end, asof, label_end) in expected.items():
        train, predict = train_rolling.build_month_configs(
            config, calendar, pd.Period(month, freq="M")
        )
        segments = train["dataset"]["kwargs"]["segments"]
        assert tuple(segments["train"]) == (start, end)
        assert "valid" not in segments
        handler = train["data_handler"]["kwargs"]
        assert handler["fit_start_time"] == start
        assert handler["fit_end_time"] == end
        assert handler["end_time"] == asof
        assert calendar.index(end) + 5 == calendar.index(asof)
        prediction_handler = predict["data_handler"]["kwargs"]
        assert prediction_handler["end_time"] == label_end
        assert prediction_handler["start_time"] == handler["start_time"]

    assert config == original


def test_partial_prediction_month_respects_requested_dates(config, calendar):
    config["dataset"]["kwargs"]["segments"]["test"] = ["2023-01-15", "2023-02-10"]
    _, january = train_rolling.build_month_configs(
        config, calendar, pd.Period("2023-01", freq="M")
    )
    _, february = train_rolling.build_month_configs(
        config, calendar, pd.Period("2023-02", freq="M")
    )

    assert tuple(january["dataset"]["kwargs"]["segments"]["test"]) == (
        "2023-01-15",
        "2023-01-31",
    )
    assert tuple(february["dataset"]["kwargs"]["segments"]["test"]) == (
        "2023-02-01",
        "2023-02-10",
    )
    assert february["data_handler"]["kwargs"]["end_time"] == "2023-02-17"


def test_run_rolling_combines_year_and_evaluates_once(
    config, calendar, tmp_path, monkeypatch
):
    calls = []
    evaluations = []

    def run_month(train, predict, data_dir, save_dir, seeds, data_seed, logger):
        calls.append((train, predict, list(seeds)))
        start, end = predict["dataset"]["kwargs"]["segments"]["test"]
        dates = [day for day in calendar if start <= day <= end]
        return pd.DataFrame(
            {
                "Instrument": ["stock"] * len(dates),
                "Date": dates,
                "RET_5D": 0.01,
                "PRED_RET_5D": 0.1,
            }
        )

    class Evaluator:
        def evaluate(self, predictions):
            evaluations.append(predictions.copy())
            return "rolling metrics"

    def init_evaluator(evaluator_config, **kwargs):
        assert evaluator_config["class"] == "Evaluator"
        assert evaluator_config["kwargs"]["start_time"] == "2023-01-01"
        assert evaluator_config["kwargs"]["end_time"] == "2023-12-31"
        return Evaluator()

    monkeypatch.setattr(train_rolling, "run_month", run_month)
    monkeypatch.setattr(
        train_rolling.DataLoader, "load_calendar", lambda *args, **kwargs: calendar
    )
    monkeypatch.setattr(train_rolling, "init_instance_by_config", init_evaluator)
    predictions = train_rolling.run_rolling(
        config,
        tmp_path / "data",
        tmp_path,
        [1234, 2024],
        1234,
        logging.getLogger("rolling-test"),
    )

    assert len(calls) == 12
    assert len(evaluations) == 1
    assert len(predictions) == len(
        [day for day in calendar if "2023-01-01" <= day <= "2023-12-31"]
    )
    assert not predictions.duplicated(["Instrument", "Date"]).any()
    assert pd.to_datetime(predictions["Date"]).dt.to_period("M").nunique() == 12
    pd.testing.assert_frame_equal(predictions, evaluations[0])
    saved = pd.read_csv(tmp_path / "predictions.csv")
    pd.testing.assert_frame_equal(saved, predictions, check_dtype=False)
    assert (tmp_path / "metrics.txt").read_text().strip() == "rolling metrics"
    assert all(seeds == [1234, 2024] for _, _, seeds in calls)


def test_run_rolling_rejects_duplicate_prediction_keys(
    config, calendar, tmp_path, monkeypatch
):
    config["dataset"]["kwargs"]["segments"]["test"] = ["2023-01-01", "2023-02-28"]
    duplicate = pd.DataFrame(
        {
            "Instrument": ["stock"],
            "Date": ["2023-01-03"],
            "RET_5D": [0.01],
            "PRED_RET_5D": [0.1],
        }
    )
    monkeypatch.setattr(
        train_rolling, "run_month", lambda *args, **kwargs: duplicate.copy()
    )
    monkeypatch.setattr(
        train_rolling.DataLoader, "load_calendar", lambda *args, **kwargs: calendar
    )

    with pytest.raises(ValueError):
        train_rolling.run_rolling(
            config,
            tmp_path / "data",
            tmp_path,
            [1234],
            1234,
            logging.getLogger("rolling-test"),
        )


def test_run_month_trains_without_validation_and_reloads_preprocessor(
    config, calendar, tmp_path, monkeypatch
):
    train, predict = train_rolling.build_month_configs(
        config, calendar, pd.Period("2023-01", freq="M")
    )
    setups = []
    fitted = []
    loaded = []

    class Handler:
        feature_names = ["TS_TEST"]

        def setup_data(self, mode):
            setups.append(mode)
            date = "2022-12-23" if mode == "train" else "2023-01-03"
            index = pd.MultiIndex.from_tuples(
                [(date, "stock")], names=["Date", "Instrument"]
            )
            return pd.DataFrame({"TS_TEST": [0.0], "RET_5D": [0.1]}, index=index)

        def save(self, path):
            Path(path).write_text("training statistics")

        def load(self, path):
            assert Path(path).read_text() == "training statistics"
            loaded.append(path)

    class Dataset:
        def __init__(self, split, feature_names):
            self.split = split
            self.feature_names = feature_names
            self.label_names = ["RET_5D"]

        def __len__(self):
            return 1

    def initialize(component, **kwargs):
        if component["class"] == "MarketAlpha158":
            return Handler()
        assert component["class"] == "TSDataset"
        assert kwargs["split"] in {"train", "test"}
        return Dataset(kwargs["split"], kwargs["feature_names"])

    def train_model(train_dataset, val_dataset, save_dir, seed, logger):
        assert train_dataset.split == "train"
        assert val_dataset is None
        fitted.append(seed)
        (save_dir / ("model_seed_%s.pth" % seed)).write_text("model")

    def ensemble(dataset, model_dir, model_names, logger):
        assert dataset.split == "test"
        assert len(loaded) == 1
        assert list(model_names) == ["model_seed_1234.pth", "model_seed_2024.pth"]
        assert all((model_dir / name).is_file() for name in model_names)
        return pd.DataFrame(
            {
                "Instrument": ["stock"],
                "Date": ["2023-01-03"],
                "RET_5D": [0.1],
                "PRED_RET_5D": [0.2],
            }
        )

    monkeypatch.setattr(train_rolling, "init_instance_by_config", initialize)
    monkeypatch.setattr(train_rolling, "train_and_save_model", train_model)
    monkeypatch.setattr(train_rolling, "ensemble_predict", ensemble)
    predictions = train_rolling.run_month(
        train,
        predict,
        tmp_path / "data",
        tmp_path / "2023-01",
        [1234, 2024],
        1234,
        logging.getLogger("rolling-test"),
    )

    assert setups == ["train", "test"]
    assert fitted == [1234, 2024]
    assert len(predictions) == 1
    assert (tmp_path / "2023-01" / "predictions.csv").is_file()
