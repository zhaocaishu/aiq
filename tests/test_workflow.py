import copy
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from aiq.utils import workflow
from tools import test as evaluation
from tools import train


LOGGER = logging.getLogger("workflow-test")


@pytest.fixture
def predictions():
    return pd.DataFrame(
        {
            "Instrument": ["b", "a", "c"],
            "Date": ["2022-01-04", "2022-01-03", "2022-01-04"],
            "RET_5D": [0.1, np.nan, -0.2],
            "PRED_RET_5D": [0.25, 0.5, -0.125],
        }
    )


@pytest.fixture
def config():
    return {
        "data_handler": {"class": "Handler", "kwargs": {}},
        "dataset": {
            "class": "Dataset",
            "kwargs": {
                "segments": {
                    "train": ["2021-01-01", "2021-12-31"],
                    "valid": ["2022-01-01", "2022-12-31"],
                    "test": ["2023-01-01", "2023-12-31"],
                },
                "label_names": ["RET_5D"],
            },
        },
        "model": {"class": "Model", "kwargs": {"learning_rate": 0.001}},
        "evaluator": {
            "class": "Evaluator",
            "kwargs": {"start_time": "2023-01-01", "end_time": "2023-12-31"},
        },
    }


def mock_predictions(monkeypatch, frames):
    def load_model(model_config, dataset, model_dir, model_name, logger):
        assert model_config == {"class": "Model", "kwargs": {}}
        return SimpleNamespace(predict=lambda data: frames[model_name].copy())

    monkeypatch.setattr(workflow, "load_model", load_model)


def predict_ensemble(tmp_path):
    return workflow.ensemble_predict(
        {"class": "Model", "kwargs": {}},
        SimpleNamespace(label_names=["RET_5D"]),
        tmp_path,
        ["first", "second"],
        LOGGER,
    )


def test_ensemble_aligns_keys_and_preserves_raw_labels(
    predictions, tmp_path, monkeypatch
):
    original = predictions.copy(deep=True)
    second = predictions.iloc[::-1].copy()
    second["PRED_RET_5D"] += 2
    mock_predictions(monkeypatch, {"first": predictions, "second": second})

    result = predict_ensemble(tmp_path)

    expected = predictions.copy()
    expected["PRED_RET_5D"] += 1
    pd.testing.assert_frame_equal(result, expected, check_exact=True)
    pd.testing.assert_frame_equal(predictions, original, check_exact=True)


@pytest.mark.parametrize(
    "failure, message",
    [
        ("labels", "labels differ"),
        ("keys", "sample keys differ"),
        ("duplicate", "duplicate Instrument/Date"),
        ("nan", "NaN or infinity"),
        ("inf", "NaN or infinity"),
        ("extra_prediction", "prediction columns do not match"),
    ],
)
def test_ensemble_rejects_inconsistent_predictions(
    predictions, tmp_path, monkeypatch, failure, message
):
    second = predictions.copy()
    if failure == "labels":
        second.loc[0, "RET_5D"] += 0.01
    elif failure == "keys":
        second.loc[0, "Instrument"] = "other"
    elif failure == "duplicate":
        second = pd.concat([second, second.iloc[[0]]], ignore_index=True)
    elif failure in {"nan", "inf"}:
        second.loc[0, "PRED_RET_5D"] = float(failure)
    else:
        second["PRED_OTHER"] = 0.0
    mock_predictions(monkeypatch, {"first": predictions, "second": second})

    with pytest.raises(ValueError, match=message):
        predict_ensemble(tmp_path)


def test_save_predictions_uses_stable_columns_and_key_order(predictions, tmp_path):
    original = predictions.copy()
    predictions["EXTRA"] = 1.0

    result = workflow.save_predictions(predictions, ["RET_5D"], tmp_path)

    expected = original.sort_values(["Date", "Instrument"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(result, expected, check_exact=True)
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "predictions.csv"), expected)
    assert "EXTRA" in predictions


def test_prediction_csv_preserves_float32_labels_and_float64_scores(tmp_path):
    predictions = pd.DataFrame(
        {
            "Instrument": ["000001", "000002", "000003"],
            "Date": ["2023-01-03"] * 3,
            "RET_5D": np.array([0.1, 1.0 / 3, np.nan], dtype=np.float32),
            "PRED_RET_5D": np.array(
                [0.12345678901234567, np.nextafter(0.1, 1), -np.pi], dtype=np.float64
            ),
        }
    )

    workflow.save_predictions(predictions, ["RET_5D"], tmp_path)
    restored = workflow.load_predictions(tmp_path / "predictions.csv", ["RET_5D"])

    expected = predictions.astype({"RET_5D": np.float64, "PRED_RET_5D": np.float64})
    pd.testing.assert_frame_equal(restored, expected, check_exact=True)
    assert predictions["RET_5D"].dtype == np.float32


def test_prediction_uses_saved_statistics_and_raw_labels(
    config, predictions, tmp_path, monkeypatch
):
    original = copy.deepcopy(config)
    events = []
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "data_handler.pkl").write_text("training statistics")
    (model_dir / "model_seed_1.pth").write_text("model")

    class Handler:
        feature_names = ["CS_TEST"]

        def load(self, path):
            assert Path(path).read_text() == "training statistics"
            events.append("load")

        def setup_data(self, mode):
            assert events == ["load"]
            assert mode == "test"
            events.append("inference")
            return pd.DataFrame({"CS_TEST": [0.0], "RET_5D": [0.1]})

    class Dataset:
        label_names = ["RET_5D"]
        feature_names = ["CS_TEST"]

        def __len__(self):
            return 1

    def initialize(component, **kwargs):
        assert component["class"] == "Dataset"
        assert kwargs["split"] == "valid"
        assert kwargs["data"]["RET_5D"].tolist() == [0.1]
        events.append("dataset")
        return Dataset()

    def ensemble(model_config, dataset, model_dir, model_names, logger):
        assert model_config == config["model"]
        assert model_names == ["model_seed_1.pth"]
        events.append("predict")
        return predictions.copy()

    monkeypatch.setattr(
        evaluation, "init_instance_by_config", lambda *a, **k: Handler()
    )
    monkeypatch.setattr(workflow, "init_instance_by_config", initialize)
    monkeypatch.setattr(workflow, "ensemble_predict", ensemble)

    result = evaluation.run_prediction(
        config,
        tmp_path / "data",
        model_dir,
        ["model_seed_1.pth"],
        "valid",
        42,
        LOGGER,
    )

    pd.testing.assert_frame_equal(result, predictions, check_exact=True)
    assert events == ["load", "inference", "dataset", "predict"]
    assert config == original


def test_csv_evaluation_needs_no_models_and_uses_selected_interval(
    config, predictions, tmp_path, monkeypatch
):
    config.pop("model")
    config.pop("data_handler")
    original = copy.deepcopy(config)
    expected = predictions.copy()
    expected["Instrument"] = ["000002", "000001", "000003"]
    pred_file = tmp_path / "scores.csv"
    expected.to_csv(pred_file, index=False)
    evaluated = []

    class Evaluator:
        def evaluate(self, frame):
            pd.testing.assert_frame_equal(frame, expected, check_exact=True)
            evaluated.append(frame.copy())
            return "validation metrics"

    def initialize(component, **kwargs):
        # Standalone evaluation must not construct a handler, dataset or model.
        assert component["class"] == "Evaluator"
        assert component["kwargs"] == {
            "start_time": "2022-01-01",
            "end_time": "2022-12-31",
        }
        return Evaluator()

    monkeypatch.setattr(workflow, "init_instance_by_config", initialize)
    metrics = evaluation.run_evaluation(
        config,
        tmp_path / "data",
        tmp_path / "evaluation",
        pred_file,
        "valid",
        LOGGER,
    )

    assert metrics == "validation metrics"
    assert len(evaluated) == 1
    assert evaluated[0]["RET_5D"].isna().sum() == 1
    assert config == original
    assert (tmp_path / "evaluation" / "metrics.txt").read_text() == metrics
    assert not list(tmp_path.glob("*.pth"))
    assert not (tmp_path / "data_handler.pkl").exists()


@pytest.mark.parametrize(
    "failure",
    ["duplicate_date", "missing_label", "missing_score", "nan", "inf", "date"],
)
def test_csv_evaluation_rejects_invalid_predictions(
    config, predictions, tmp_path, monkeypatch, failure
):
    invalid = predictions.copy()
    if failure == "duplicate_date":
        duplicate = invalid.iloc[[0]].copy()
        duplicate["Date"] = "2022-01-04 00:00:00"
        invalid = pd.concat([invalid, duplicate], ignore_index=True)
    elif failure == "missing_label":
        invalid = invalid.drop(columns="RET_5D")
    elif failure == "missing_score":
        invalid = invalid.drop(columns="PRED_RET_5D")
    elif failure == "date":
        invalid.loc[0, "Date"] = "not a date"
    else:
        invalid.loc[0, "PRED_RET_5D"] = float(failure)
    pred_file = tmp_path / "scores.csv"
    invalid.to_csv(pred_file, index=False)

    def initialize(*args, **kwargs):
        pytest.fail("Invalid CSV must be rejected before initializing the evaluator")

    monkeypatch.setattr(workflow, "init_instance_by_config", initialize)
    with pytest.raises(ValueError):
        evaluation.run_evaluation(
            config,
            tmp_path / "data",
            tmp_path / "evaluation",
            pred_file,
            "valid",
            LOGGER,
        )


def test_training_prepares_once_and_seeds_each_model(config, tmp_path, monkeypatch):
    original = copy.deepcopy(config)
    events = []
    seeds = []
    prepared = pd.DataFrame({"CS_TEST": [0.0], "RET_5D": [0.1]})

    class Handler:
        feature_names = ["CS_TEST"]

        def setup_data(self, mode):
            assert mode == "train"
            assert seeds == [42]
            events.append("prepare")
            return prepared

        def save(self, path):
            events.append("save_handler")
            Path(path).write_text("training statistics")

    class Dataset:
        label_names = ["RET_5D"]
        feature_names = ["CS_TEST"]

        def __init__(self, split):
            self.split = split

        def __len__(self):
            return 1

    class Model:
        def fit(self, train_dataset, val_dataset):
            assert train_dataset.split == "train"
            assert val_dataset.split == "valid"
            assert "save_handler" in events
            events.append(("fit", seeds[-1]))

        def save(self, model_name):
            events.append(("save_model", model_name))
            (tmp_path / model_name).write_text("model")

    def initialize(component, **kwargs):
        if component["class"] == "Dataset":
            assert kwargs["data"] is prepared
            events.append(kwargs["split"])
            return Dataset(kwargs["split"])
        assert component == config["model"]
        assert kwargs["feature_names"] == ["CS_TEST"]
        return Model()

    monkeypatch.setattr(train, "init_instance_by_config", lambda *a, **k: Handler())
    monkeypatch.setattr(workflow, "init_instance_by_config", initialize)
    monkeypatch.setattr(workflow, "set_random_seed", seeds.append)

    names = train.run_training(config, tmp_path / "data", tmp_path, [1, 2], 42, LOGGER)

    assert seeds == [42, 1, 2]
    assert names == ["model_seed_1.pth", "model_seed_2.pth"]
    assert events == [
        "prepare",
        "train",
        "valid",
        "save_handler",
        ("fit", 1),
        ("save_model", names[0]),
        ("fit", 2),
        ("save_model", names[1]),
    ]
    assert config == original


def test_evaluation_rejects_predictions_outside_interval(config, predictions, tmp_path):
    with pytest.raises(ValueError):
        workflow.evaluate_predictions(
            config["evaluator"],
            predictions,
            tmp_path,
            tmp_path,
            ["2023-01-01", "2023-12-31"],
            LOGGER,
        )


def test_testing_cli_predicts_saves_and_evaluates_without_training(
    config, predictions, tmp_path, monkeypatch
):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config))
    model_dir = tmp_path / "existing_models"
    output_dir = tmp_path / "evaluation"
    calls = []
    expected = predictions.sort_values(["Date", "Instrument"]).reset_index(drop=True)

    def fail_training(*args, **kwargs):
        pytest.fail("Testing must not train or save a new handler")

    def run_prediction(
        prediction_config,
        data_dir,
        actual_model_dir,
        model_names,
        split,
        data_seed,
        logger,
    ):
        assert prediction_config == config
        assert actual_model_dir == model_dir
        assert model_names == ["first.pth", "second.pth"]
        assert split == "valid"
        assert data_seed == 42
        calls.append("predict")
        return predictions.copy()

    class Evaluator:
        def evaluate(self, frame):
            assert calls == ["predict"]
            assert (output_dir / "predictions.csv").is_file()
            pd.testing.assert_frame_equal(frame, expected, check_exact=True)
            calls.append("evaluate")
            return "validation metrics"

    def initialize(component, **kwargs):
        assert component["class"] == "Evaluator"
        assert component["kwargs"] == {
            "start_time": "2022-01-01",
            "end_time": "2022-12-31",
        }
        return Evaluator()

    monkeypatch.setattr(train, "run_training", fail_training)
    monkeypatch.setattr(workflow, "save_handler", fail_training)
    monkeypatch.setattr(workflow, "init_instance_by_config", initialize)
    monkeypatch.setattr(evaluation, "run_prediction", run_prediction)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test.py",
            "--cfg_file",
            str(config_file),
            "--data_dir",
            str(tmp_path / "data"),
            "--save_dir",
            str(output_dir),
            "--model_dir",
            str(model_dir),
            "--model_names",
            "first.pth",
            "second.pth",
            "--split",
            "valid",
            "--data_seed",
            "42",
        ],
    )

    evaluation.main()

    assert calls == ["predict", "evaluate"]
    assert (output_dir / "metrics.txt").read_text() == "validation metrics"
    assert not (output_dir / "data_handler.pkl").exists()


def test_existing_csv_cli_does_not_load_or_predict_models(
    config, predictions, tmp_path, monkeypatch
):
    config.pop("model")
    config.pop("data_handler")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config))
    pred_file = tmp_path / "existing.csv"
    predictions.to_csv(pred_file, index=False, float_format="%.17g")
    calls = []

    def fail_prediction(*args, **kwargs):
        pytest.fail(
            "Existing prediction CSV must not trigger model loading or inference"
        )

    class Evaluator:
        def evaluate(self, frame):
            pd.testing.assert_frame_equal(frame, predictions, check_exact=True)
            calls.append("evaluate")
            return "existing prediction metrics"

    def initialize(component, **kwargs):
        assert component["class"] == "Evaluator"
        return Evaluator()

    monkeypatch.setattr(evaluation, "run_test", fail_prediction)
    monkeypatch.setattr(evaluation, "run_prediction", fail_prediction)
    monkeypatch.setattr(workflow, "init_instance_by_config", initialize)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test.py",
            "--cfg_file",
            str(config_file),
            "--data_dir",
            str(tmp_path / "data"),
            "--save_dir",
            str(tmp_path / "evaluation"),
            "--pred_file",
            str(pred_file),
            "--split",
            "valid",
        ],
    )

    evaluation.main()

    assert calls == ["evaluate"]
    assert not list(tmp_path.glob("*.pth"))
    assert not (tmp_path / "data_handler.pkl").exists()


def test_training_cli_only_trains(config, tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config))
    calls = []

    def run_training(training_config, data_dir, save_dir, seeds, data_seed, logger):
        assert training_config == config
        assert save_dir == tmp_path
        assert seeds == [17, 23]
        assert data_seed == 42
        calls.append("train")
        return ["model_seed_17.pth", "model_seed_23.pth"]

    def fail_prediction(*args, **kwargs):
        pytest.fail("Training entry must not run prediction or evaluation")

    monkeypatch.setattr(train, "run_training", run_training)
    monkeypatch.setattr(workflow, "ensemble_predict", fail_prediction)
    monkeypatch.setattr(workflow, "evaluate_predictions", fail_prediction)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--cfg_file",
            str(config_file),
            "--data_dir",
            str(tmp_path / "data"),
            "--save_dir",
            str(tmp_path),
            "--seeds",
            "17",
            "23",
            "--data_seed",
            "42",
        ],
    )

    train.main()

    assert calls == ["train"]
    assert not (tmp_path / "predictions.csv").exists()
    assert not (tmp_path / "metrics.txt").exists()


def test_testing_cli_defaults_to_checkpoint_prediction(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test.py",
            "--cfg_file",
            str(tmp_path / "config.yaml"),
            "--data_dir",
            str(tmp_path),
            "--save_dir",
            str(tmp_path),
        ],
    )

    args = evaluation.parse_args()

    assert args.pred_file is None
    assert args.model_dir == tmp_path
    assert args.model_names == ["model_seed_1234.pth"]
    assert args.split == "test"


@pytest.mark.parametrize("model_argument", ["--model_dir", "--model_names"])
def test_testing_cli_rejects_csv_and_checkpoint_arguments(
    tmp_path, monkeypatch, model_argument
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test.py",
            "--cfg_file",
            str(tmp_path / "config.yaml"),
            "--data_dir",
            str(tmp_path),
            "--save_dir",
            str(tmp_path),
            "--pred_file",
            str(tmp_path / "existing.csv"),
            model_argument,
            "model",
        ],
    )

    with pytest.raises(SystemExit) as error:
        evaluation.parse_args()

    assert error.value.code == 2
