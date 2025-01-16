import pandas as pd

from aiq.dataset.processor import CSZScoreNorm


if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "Date": ["2022-01-01", "2022-01-01", "2022-01-01", "2022-01-01"],
            "Instrument": ["a", "b", "b", "a"],
            "Factor_0": [0.1, 2.4, 1.2, None],
            "Factor_1": [2.0, 1.0, 3.0, 4.0],
        }
    )
    df.set_index(["Date", "Instrument"], inplace=True)
    df.columns = pd.MultiIndex.from_tuples(
        [("feature", "Factor_0"), ("feature", "Factor_1")]
    )
    print(df)

    cszscore_norm = CSZScoreNorm(fields_group='feature')
    print(cszscore_norm(df))
