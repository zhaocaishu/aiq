import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model")

    parser.add_argument(
        "--factor_file",
        type=str,
        default=None,
        help="Path to the factor file for analysis.",
    )

    parser.add_argument(
        "--factor_names",
        type=str,
        default=None,
        help="Factor names.",
    )

    return parser.parse_args()


def corr_analysis(factor_df, factor_names):
    factor_df = factor_df[factor_names]
    correlation_matrix = factor_df.corr()

    # Find pairs with correlation > 0.9
    high_corr_pairs = []
    for i in range(len(factor_names)):
        for j in range(i + 1, len(factor_names)):
            if correlation_matrix.iloc[i, j] > 0.9:
                high_corr_pairs.append((factor_names[i], factor_names[j]))

    # Print the pairs
    for pair in high_corr_pairs:
        print(
            f"{pair[0]} 和 {pair[1]} 的相关性为 {correlation_matrix.loc[pair[0], pair[1]]:.4f}"
        )

    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.show()


def main():
    args = parse_args()

    factor_names = args.factor_names.split(",")
    factor_df = pd.read_csv(args.factor_file)
    corr_analysis(factor_df=factor_df, factor_names=factor_names)


if __name__ == "__main__":
    main()
