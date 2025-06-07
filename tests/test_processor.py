import pandas as pd

from aiq.dataset.processor import CSZScoreNorm, CSNeutralize


if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "Date": ["2022-01-01", "2022-01-01", "2022-01-01", "2022-01-01"],
            "Instrument": ["a", "b", "c", "d"],
            "IND_CLS_CAT": [0, 0, 0, 0],
            "MKT_CAP": [2.5, 5.2, 3.0, 10.0],
            "Factor_0": [0.1, 2.4, 1.2, None],
            "Factor_1": [2.0, 1.0, 3.0, 4.0],
        }
    )
    df.set_index(["Date", "Instrument"], inplace=True)
    df.columns = pd.MultiIndex.from_tuples(
        [
            ("feature", "IND_CLS_CAT"),
            ("feature", "MKT_CAP"),
            ("feature", "Factor_0"),
            ("feature", "Factor_1"),
        ]
    )
    print(df)

    cs_neutralize = CSNeutralize(
        industry_col="IND_CLS_CAT", cap_col="MKT_CAP", factor_cols=["Factor_0", "Factor_1"]
    )
    print(cs_neutralize(df))

    cszscore_norm = CSZScoreNorm(fields_group="feature")
    print(cszscore_norm(df))
