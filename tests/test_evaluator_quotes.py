import numpy as np
import pandas as pd
import pytest

from aiq.dataset.loader import DataLoader
from aiq.evaluation import Evaluator


@pytest.fixture
def quote_inputs(monkeypatch):
    dates = ["2023-01-03", "2023-01-04", "2023-01-05"]
    quotes = pd.DataFrame(
        [
            {
                "Date": date,
                "Instrument": instrument,
                "Close": 10.0,
                "High": 10.0,
                "Low": 10.0,
                "Volume": 100.0,
                "AMount": 100.0,
                "Adj_factor": 1.0,
                "Up_limit": 11.0,
                "Down_limit": 9.0,
            }
            for date in dates
            for instrument in ("A", "B")
        ]
    )
    benchmark = pd.DataFrame({"Date": dates, "Instrument": "INDEX", "Close": 10.0})
    predictions = quotes[["Date", "Instrument"]].copy()
    predictions["PRED_RET_5D"] = [2.0, 1.0, 1.0, 2.0, 1.0, 2.0]
    predictions["RET_5D"] = 0.0
    monkeypatch.setattr(DataLoader, "load_calendar", lambda *args, **kwargs: dates)
    monkeypatch.setattr(
        DataLoader,
        "load_instruments",
        lambda *args, **kwargs: quotes[["Date", "Instrument"]].copy(),
    )
    monkeypatch.setattr(
        DataLoader, "load_instruments_features", lambda *args, **kwargs: quotes.copy()
    )
    monkeypatch.setattr(
        DataLoader, "load_markets_features", lambda *args, **kwargs: benchmark.copy()
    )
    evaluator = Evaluator(
        data_dir="",
        start_time=dates[0],
        end_time=dates[-1],
        top_k=1,
        risk_analysis=False,
    )
    return evaluator, quotes, predictions


@pytest.mark.parametrize(
    "volume, amount",
    [
        (0.0, 1000.0),
        (100.0, 0.0),
        (-1.0, 100.0),
        (100.0, -1.0),
        (np.nan, 100.0),
        (100.0, np.inf),
        (np.inf, 100.0),
        (100.0, "invalid"),
        (1e-320, 1000.0),
        (1e308, np.nextafter(0.0, 1.0)),
    ],
)
def test_evaluate_rejects_invalid_trade_data(quote_inputs, volume, amount):
    evaluator, quotes, predictions = quote_inputs
    # A would otherwise be sold on this session at the corrupted VWAP.
    row = quotes["Date"].eq("2023-01-05") & quotes["Instrument"].eq("A")
    quotes.loc[row, ["Volume", "AMount"]] = [volume, amount]

    with pytest.raises(ValueError, match="2023-01-05 A.*VWAP"):
        evaluator.evaluate(predictions)


def test_evaluate_rejects_nullable_missing_volume(quote_inputs):
    evaluator, quotes, predictions = quote_inputs
    quotes["Volume"] = quotes["Volume"].astype("Float64")
    row = quotes["Date"].eq("2023-01-05") & quotes["Instrument"].eq("A")
    quotes.loc[row, "Volume"] = pd.NA

    with pytest.raises(ValueError, match="2023-01-05 A.*VWAP"):
        evaluator.evaluate(predictions)


def test_zero_volume_session_has_no_execution_price(quote_inputs, monkeypatch):
    evaluator, quotes, predictions = quote_inputs
    row = quotes["Date"].eq("2023-01-05") & quotes["Instrument"].eq("A")
    quotes.loc[row, ["Volume", "AMount"]] = 0.0
    snapshots = []
    monkeypatch.setattr(evaluator.backtester, "_log_daily_holdings", snapshots.append)

    prepared = evaluator._prepare_dataset(predictions)
    assert prepared.loc[row, "VWAP"].isna().all()
    evaluator.evaluate(predictions)

    assert snapshots[-1].sell_value == 0
    assert [holding.instrument for holding in snapshots[-1].holdings] == ["A"]
    assert np.isfinite(snapshots[-1].total_asset)


def test_valid_trade_data_retains_vwap_units(quote_inputs):
    evaluator, quotes, predictions = quote_inputs
    quotes["Volume"] = "200"
    quotes["AMount"] = "199"

    prepared = evaluator._prepare_dataset(predictions)

    assert prepared["VWAP"].tolist() == pytest.approx([9.95] * len(quotes))
