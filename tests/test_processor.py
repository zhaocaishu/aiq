import pandas as pd

from aiq.dataset.processor import CSZScoreNorm


if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "Date": ["2022-01-01", "2022-01-01", "2022-01-01", "2022-01-01"],
            "Instrument": ["a", "b", "b", "a"],
            "Ind_class": [1, 2, 1, 1],
            "Market_cap": [3, 5, 4, 100],
            "Factor_0": [0.1, 2.4, 1.2, None],
            "Factor_1": [2.0, 1.0, 3.0, 4.0],
        }
    )
    df.set_index(["Date", "Instrument"], inplace=True)
    print(df)

    cszscore_norm = CSZScoreNorm(target_cols=["Factor_0", "Factor_1"])
    print(cszscore_norm(df))
