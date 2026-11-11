from pathlib import Path

import pandas as pd
import pytest

from aiq.evaluation import Evaluator
from aiq.dataset.loader import DataLoader


def test_evaluate_uses_adjustment_factor_for_portfolio(monkeypatch):
    data_dir = Path(__file__).resolve().parents[1] / "data"
    evaluator = Evaluator(
        data_dir=str(data_dir),
        start_time="2023-01-03",
        end_time="2023-01-04",
        benchmark="000906.SH",
        top_k=1,
        risk_analysis=False,
    )
    predictions = pd.DataFrame(
        {
            "Date": ["2023-01-03"],
            "Instrument": ["000990.SZ"],
            "PRED_RET_5D": [0.1],
            "RET_5D": [0.02],
        }
    )
    snapshots = []
    monkeypatch.setattr(evaluator.backtester, "_log_daily_holdings", snapshots.append)

    evaluator.evaluate(predictions)

    (holding,) = snapshots[-1].holdings
    assert holding.instrument == "000990.SZ"
    assert holding.adjusted_close == pytest.approx(8.95 * 3.12)


@pytest.fixture
def suspended_market(monkeypatch):
    dates = ["2023-01-03", "2023-01-04", "2023-01-05"]
    universe = pd.DataFrame(
        [(date, instrument) for date in dates for instrument in ("A", "B")],
        columns=["Date", "Instrument"],
    )
    quotes = universe.loc[
        ~((universe["Date"] == dates[1]) & (universe["Instrument"] == "A"))
    ].assign(
        Close=10.0,
        High=10.0,
        Low=10.0,
        Volume=100.0,
        AMount=100.0,
        Adj_factor=1.0,
        Up_limit=11.0,
        Down_limit=9.0,
    )
    benchmark = pd.DataFrame({"Date": dates, "Instrument": "INDEX", "Close": 10.0})
    monkeypatch.setattr(DataLoader, "load_calendar", lambda *args, **kwargs: dates)
    monkeypatch.setattr(
        DataLoader, "load_instruments", lambda *args, **kwargs: universe.copy()
    )
    monkeypatch.setattr(
        DataLoader, "load_instruments_features", lambda *args, **kwargs: quotes.copy()
    )
    monkeypatch.setattr(
        DataLoader, "load_markets_features", lambda *args, **kwargs: benchmark.copy()
    )
    predictions = pd.DataFrame(
        [
            (dates[0], "A", 3.0),
            (dates[0], "B", 2.0),
            (dates[1], "A", 4.0),
            (dates[1], "B", 1.0),
        ],
        columns=["Date", "Instrument", "PRED_RET_5D"],
    ).assign(RET_5D=0.0)
    return dates, quotes, predictions


@pytest.mark.parametrize("risk_analysis", [False, True])
@pytest.mark.parametrize("has_resume_signal", [False, True])
def test_resume_uses_previous_session_signal(
    monkeypatch, suspended_market, risk_analysis, has_resume_signal
):
    dates, _, predictions = suspended_market
    if not has_resume_signal:
        predictions = predictions.loc[
            ~((predictions["Date"] == dates[1]) & (predictions["Instrument"] == "A"))
        ]
    evaluator = Evaluator(
        "unused", dates[0], dates[-1], top_k=1, risk_analysis=risk_analysis
    )
    snapshots = []
    monkeypatch.setattr(evaluator.backtester, "_log_daily_holdings", snapshots.append)

    evaluator.evaluate(predictions)

    assert [holding.instrument for holding in snapshots[1].holdings] == ["B"]
    expected = "A" if has_resume_signal else "B"
    assert [holding.instrument for holding in snapshots[2].holdings] == [expected]


def test_signals_do_not_hide_a_missing_market_day(monkeypatch, suspended_market):
    dates, quotes, predictions = suspended_market
    quotes.drop(quotes.index[quotes["Date"] == dates[1]], inplace=True)
    evaluator = Evaluator("unused", dates[0], dates[-1], top_k=1, risk_analysis=False)

    with pytest.raises(ValueError, match="2023-01-04.*缺少整日行情"):
        evaluator.evaluate(predictions)
