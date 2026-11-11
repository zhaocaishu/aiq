import pandas as pd
import pytest

from aiq.evaluation.backtester import TopKDropoutBacktester


def _run_backtest(monkeypatch, candidates, initial_holdings):
    dates = ["2023-01-03", "2023-01-04", "2023-01-05"]
    rows = []
    for date in dates:
        for instrument, (score, final_price) in candidates.items():
            price = final_price if date == dates[-1] else 10
            rows.append(
                {
                    "Date": date,
                    "Instrument": instrument,
                    "PRED_RET_5D": (
                        int(instrument in initial_holdings)
                        if date == dates[0]
                        else score
                    ),
                    "Close": price,
                    "VWAP": price,
                    "Adj_factor": 1,
                    "Up_limit": price * 1.1,
                    "Down_limit": price * 0.9,
                }
            )

    snapshots = []
    backtester = TopKDropoutBacktester(
        top_k=len(initial_holdings), trading_calendar=dates
    )
    monkeypatch.setattr(backtester, "_log_daily_holdings", snapshots.append)
    backtester.run(pd.DataFrame(rows), initial_capital=10_000)
    return snapshots[1:]


@pytest.mark.parametrize(
    "candidates, final_holding, trades",
    [
        pytest.param({"A": (2, 10)}, "A", (0, 0, 995), id="empty-buy-pool"),
        pytest.param(
            {"A": (float("nan"), 10), "B": (float("nan"), 10)},
            "A",
            (0, 0, 995),
            id="missing-scores",
        ),
        pytest.param(
            {"A": (2, 10), "B": (3, 96)},
            "A",
            (0, 0, 995),
            id="replacement-exceeds-target-budget",
        ),
        pytest.param(
            {"A": (2, 10), "B": (4, 96), "C": (1, 10)},
            "A",
            (0, 0, 995),
            id="affordable-fallback-ranks-below-holding",
        ),
        pytest.param(
            {"A": (1, 10), "B": (3, 96), "C": (2, 10)},
            "C",
            (9000, 9000, 976),
            id="affordable-fallback-outranks-holding",
        ),
        pytest.param(
            {"A": (1, 9), "B": (2, 90.8)},
            "A",
            (0, 0, 995),
            id="insufficient-cash-after-fees",
        ),
    ],
)
def test_dropout_sells_only_with_an_affordable_better_replacement(
    monkeypatch, candidates, final_holding, trades
):
    opening, rebalance = _run_backtest(monkeypatch, candidates, {"A"})
    assert [holding.instrument for holding in opening.holdings] == ["A"]
    assert [holding.instrument for holding in rebalance.holdings] == [final_holding]
    assert (rebalance.sell_value, rebalance.buy_value, rebalance.cash) == pytest.approx(
        trades
    )


def test_dropout_reserves_cash_for_each_planned_replacement(monkeypatch):
    candidates = {"A": (2, 4), "B": (1, 4), "C": (4, 10), "D": (3, 32)}
    opening, rebalance = _run_backtest(monkeypatch, candidates, {"A", "B"})

    assert {holding.instrument for holding in opening.holdings} == {"A", "B"}
    assert {holding.instrument for holding in rebalance.holdings} == {"C", "D"}
    assert (rebalance.sell_value, rebalance.buy_value, rebalance.cash) == pytest.approx(
        (3200, 4200, 966.8)
    )


@pytest.mark.parametrize(
    "prices, top_k, risk_degree, expected_shares, expected_cash",
    [
        pytest.param(
            {"A": 100, "B": 10},
            1,
            0.95,
            {"B": 900},
            995,
            id="skip-unaffordable-best",
        ),
        pytest.param(
            {"A": 10, "B": 80, "C": 20, "D": 5},
            2,
            0.95,
            {"A": 400, "C": 200},
            1990,
            id="fill-remaining-slot-with-next-candidate",
        ),
        pytest.param(
            {"A": 100, "B": 10},
            1,
            1.0,
            {"B": 900},
            995,
            id="lot-affordable-only-before-commission",
        ),
        pytest.param(
            {"A": 10},
            3,
            0.95,
            {"A": 900},
            995,
            id="keep-budget-for-small-candidate-pool",
        ),
        pytest.param(
            {"A": 100, "B": 200},
            2,
            0.95,
            {},
            10000,
            id="no-affordable-candidate",
        ),
    ],
)
def test_opening_fills_slots_from_affordable_ranked_candidates(
    monkeypatch, prices, top_k, risk_degree, expected_shares, expected_cash
):
    dates = ["2023-01-03", "2023-01-04"]
    rows = [
        {
            "Date": date,
            "Instrument": instrument,
            "PRED_RET_5D": len(prices) - rank,
            "Close": price,
            "VWAP": price,
            "Adj_factor": 1,
            "Up_limit": price * 1.1,
            "Down_limit": price * 0.9,
        }
        for date in dates
        for rank, (instrument, price) in enumerate(prices.items())
    ]
    snapshots = []
    backtester = TopKDropoutBacktester(top_k=top_k, trading_calendar=dates)
    monkeypatch.setattr(backtester, "_log_daily_holdings", snapshots.append)

    backtester.run(pd.DataFrame(rows), initial_capital=10_000, risk_degree=risk_degree)

    opening = snapshots[-1]
    shares = {
        holding.instrument: holding.adjusted_shares for holding in opening.holdings
    }
    assert shares == expected_shares
    assert opening.cash == pytest.approx(expected_cash)
