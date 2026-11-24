import numpy as np
import pandas as pd

from aiq.dataset.dataset import CSDataset, TSDataset


def test_cs_dataset_matches_ppnet_samples_features_and_labels(tmp_path):
    days = ["2022-01-03", "2022-01-04", "2022-01-05"]
    stocks = [f"S{i:02d}" for i in range(14)]
    index = pd.MultiIndex.from_product([days, stocks], names=["Date", "Instrument"])
    value = np.tile(np.arange(len(stocks), dtype=np.float32), len(days))
    data = pd.DataFrame(
        {
            "CS_PRICE": value,
            "TS_RET": value,
            "CS_VOLUME": value**2,
            "IND_CLS_L1": 1,
            "IND_CLS_L2": 2,
            "MKT_RET": value,
            "FUND_CAP": value,
            "RET_5D": value / 100,
        },
        index=index,
        dtype=np.float32,
    )
    data = data.drop(index=(days[0], "S12"))
    data.loc[(days[1], "S10"), "CS_VOLUME"] = np.nan
    data.loc[(days[1], "S11"), ["CS_PRICE", "RET_5D"]] = [10000, np.nan]
    members = [(days[1], s) for s in stocks[:-1]] + [(days[2], s) for s in stocks[1:]]
    (tmp_path / "universes").mkdir()
    pd.DataFrame(members, columns=index.names).to_csv(
        tmp_path / "universes" / "pool.csv", index=False
    )
    kwargs = dict(
        data=data,
        segments={"valid": (days[1], days[2])},
        split="valid",
        data_dir=str(tmp_path),
        universe="pool",
        seq_len=2,
        feature_names=list(data.columns[:-1]),
        label_names=["RET_5D"],
    )
    tree = CSDataset(**kwargs)
    ppnet = TSDataset(**kwargs)
    assert tree.feature_names == ["CS_PRICE", "CS_VOLUME"]

    # S12 joins before it has two observations; S13 enters with pre-segment history.
    expected_keys = [key for key in members if key != (days[1], "S12")]
    assert set(tree.data.index) == set(expected_keys)
    pd.testing.assert_series_equal(
        tree.data["RET_5D"], data.loc[tree.data.index, "RET_5D"]
    )
    assert tree.data.loc[(days[1], "S11"), "CS_PRICE"] == 3.0
    assert tree.data.loc[(days[1], "S10"), "CS_VOLUME"] == 0.0

    expected = []
    for batch in ppnet:
        keys = ppnet.data.index[batch["sample_indices"]].swaplevel()
        frame = pd.DataFrame(
            batch["stock_cs_features"], index=keys, columns=tree.feature_names
        )
        frame["RET_5D"] = batch["labels"][:, 0]
        expected.append(frame)
    expected = pd.concat(expected).loc[tree.data.index]
    np.testing.assert_allclose(
        tree.data[tree.feature_names], expected[tree.feature_names], atol=1e-6
    )
    supervised = expected.dropna(subset=["RET_5D"])
    x, y = tree.get_training_data()
    np.testing.assert_allclose(x, supervised[tree.feature_names], atol=1e-6)
    np.testing.assert_allclose(y, supervised[["RET_5D"]], atol=1e-6)
