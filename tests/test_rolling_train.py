import copy
import logging
from pathlib import Path

import pandas as pd
import pytest
import yaml

from aiq.utils import workflow
from tools import rolling_train
from tools import test as evaluation


@pytest.fixture
def config():
    path = Path(__file__).resolve().parents[1] / "configs" / "ppnet_model_rolling.yaml"
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
        train, predict = rolling_train.build_month_configs(
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
    _, january = rolling_train.build_month_configs(
        config, calendar, pd.Period("2023-01", freq="M")
    )
    _, february = rolling_train.build_month_configs(
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


def test_run_rolling_combines_year_for_separate_evaluation(
    config, calendar, tmp_path, monkeypatch
):
    original_config = copy.deepcopy(config)
    training_config = copy.deepcopy(config)
    training_config.pop("evaluator")
    original_training_config = copy.deepcopy(training_config)
    calls = []
    evaluations = []
    extractions = []
    cached_data = {"instrument": pd.DataFrame()}

    def run_month(
        train, predict, data_dir, save_dir, seeds, data_seed, logger, raw_data
    ):
        assert raw_data is cached_data
        assert "evaluator" not in train and "evaluator" not in predict
        calls.append((train, predict, list(seeds)))
        start, end = predict["dataset"]["kwargs"]["segments"]["test"]
        dates = [day for day in reversed(calendar) if start <= day <= end]
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

    class Handler:
        def fetch_data(self):
            extractions.append(True)
            return cached_data

    def init_handler(handler_config, **kwargs):
        assert handler_config["class"] == "MarketAlpha158"
        assert handler_config["kwargs"]["start_time"] == "2016-07-01"
        assert handler_config["kwargs"]["end_time"] == "2024-01-08"
        return Handler()

    def init_evaluator(evaluator_config, **kwargs):
        assert evaluator_config["class"] == "Evaluator"
        assert evaluator_config["kwargs"]["start_time"] == "2023-01-01"
        assert evaluator_config["kwargs"]["end_time"] == "2023-12-31"
        return Evaluator()

    monkeypatch.setattr(rolling_train, "run_month", run_month)
    monkeypatch.setattr(
        rolling_train.DataLoader, "load_calendar", lambda *args, **kwargs: calendar
    )
    monkeypatch.setattr(rolling_train, "init_instance_by_config", init_handler)
    monkeypatch.setattr(workflow, "init_instance_by_config", init_evaluator)

    def fail_evaluation(*args, **kwargs):
        pytest.fail("Rolling training must only train and save predictions")

    with monkeypatch.context() as context:
        context.setattr(workflow, "evaluate_predictions", fail_evaluation)
        predictions = rolling_train.run_rolling(
            training_config,
            tmp_path / "data",
            tmp_path,
            [1234, 2024],
            1234,
            logging.getLogger("rolling-test"),
        )

    assert len(calls) == 12
    assert len(extractions) == 1
    assert evaluations == []
    expected_dates = [day for day in calendar if "2023-01-01" <= day <= "2023-12-31"]
    assert predictions["Date"].tolist() == expected_dates
    assert not predictions.duplicated(["Instrument", "Date"]).any()
    assert pd.to_datetime(predictions["Date"]).dt.to_period("M").nunique() == 12
    saved = workflow.load_predictions(tmp_path / "predictions.csv", ["RET_5D"])
    pd.testing.assert_frame_equal(saved, predictions, check_exact=True)
    assert saved["RET_5D"].eq(0.01).all()
    assert saved["PRED_RET_5D"].eq(0.1).all()
    assert not (tmp_path / "metrics.txt").exists()
    assert all(seeds == [1234, 2024] for _, _, seeds in calls)
    assert training_config == original_training_config

    metrics = evaluation.run_evaluation(
        config,
        tmp_path / "data",
        tmp_path / "evaluation",
        tmp_path / "predictions.csv",
        "test",
        logging.getLogger("rolling-test"),
    )

    assert metrics == "rolling metrics"
    assert len(evaluations) == 1
    pd.testing.assert_frame_equal(evaluations[0], predictions, check_exact=True)
    assert len(calls) == 12 and len(extractions) == 1
    assert (tmp_path / "evaluation" / "metrics.txt").read_text() == metrics
    assert not (tmp_path / "metrics.txt").exists()
    assert config == original_config


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
        rolling_train, "run_month", lambda *args, **kwargs: duplicate.copy()
    )
    monkeypatch.setattr(
        rolling_train.DataLoader, "load_calendar", lambda *args, **kwargs: calendar
    )

    class Handler:
        def fetch_data(self):
            return {}

    monkeypatch.setattr(
        rolling_train, "init_instance_by_config", lambda *args, **kwargs: Handler()
    )

    with pytest.raises(ValueError):
        rolling_train.run_rolling(
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
    train, predict = rolling_train.build_month_configs(
        config, calendar, pd.Period("2023-01", freq="M")
    )
    setups = []
    fitted = []
    loaded = []
    index = pd.MultiIndex.from_product(
        [calendar, ["stock"]], names=["Date", "Instrument"]
    )
    raw_data = {
        "instrument": pd.DataFrame({"TS_TEST": 1.0, "RET_5D": 0.1}, index=index),
        "market": pd.DataFrame(
            {"MKT_TEST": 1.0}, index=pd.Index(calendar, name="Date")
        ),
    }
    original = copy.deepcopy(raw_data)
    seed_events = []

    class Handler:
        feature_names = ["TS_TEST"]

        def __init__(self):
            self.calendar = calendar

        def process_data(self, data, mode):
            setups.append(mode)
            if mode == "train":
                first = pd.Index(calendar).searchsorted("2017-01-01")
                context_start = calendar[
                    first - config["dataset"]["kwargs"]["seq_len"] + 1
                ]
                dates = data["instrument"].index.get_level_values("Date")
                assert dates.min() == context_start
                assert dates.max() == "2022-12-23"
                assert data["market"].index.min() == context_start
                assert data["market"].index.max() == "2022-12-23"
            else:
                first = calendar.index("2023-01-03")
                context_start = calendar[
                    first - config["dataset"]["kwargs"]["seq_len"] + 1
                ]
                dates = data["instrument"].index.get_level_values("Date")
                assert dates.min() == context_start
                assert dates.max() == "2023-01-31"
                assert data["market"].index.min() == context_start
                assert data["market"].index.max() == "2023-01-31"
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

    class Model:
        def fit(self, train_dataset, val_dataset):
            assert train_dataset.split == "train"
            assert val_dataset is None
            fitted.append(seed_events[-1])

        def save(self, model_name):
            (tmp_path / "2023-01" / model_name).write_text("model")

    def initialize(component, **kwargs):
        if component["class"] == "MarketAlpha158":
            return Handler()
        if component["class"] == "PPNetModel":
            assert component == train["model"]
            return Model()
        assert component["class"] == "TSDataset"
        assert kwargs["split"] in {"train", "test"}
        return Dataset(kwargs["split"], kwargs["feature_names"])

    def ensemble(model_config, dataset, model_dir, model_names, logger):
        assert model_config == predict["model"]
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

    monkeypatch.setattr(rolling_train, "init_instance_by_config", initialize)
    monkeypatch.setattr(workflow, "init_instance_by_config", initialize)
    monkeypatch.setattr(workflow, "set_random_seed", seed_events.append)
    monkeypatch.setattr(workflow, "ensemble_predict", ensemble)
    predictions = rolling_train.run_month(
        train,
        predict,
        tmp_path / "data",
        tmp_path / "2023-01",
        [1234, 2024],
        1234,
        logging.getLogger("rolling-test"),
        raw_data,
    )

    assert setups == ["train", "test"]
    assert fitted == [1234, 2024]
    assert seed_events == [1234, 1234, 2024, 1234]
    assert len(predictions) == 1
    assert (tmp_path / "2023-01" / "predictions.csv").is_file()
    for name in raw_data:
        pd.testing.assert_frame_equal(raw_data[name], original[name])
