import sys
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a model with enhanced output clarity"
    )
    parser.add_argument(
        "--factor_file",
        type=str,
        required=True,
        help="Path to the factor file for analysis (CSV).",
    )
    parser.add_argument(
        "--factor_names",
        type=str,
        required=True,
        help="Comma-separated list of factor names, e.g., 'factor1,factor2,factor3'.",
    )
    parser.add_argument(
        "--return_col",
        type=str,
        required=True,
        help="Column name of returns/labels for IC analysis.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Threshold for high correlation detection.",
    )
    parser.add_argument(
        "--save_fig",
        type=str,
        default=None,
        help="Path to save the correlation heatmap. If not set, show the plot.",
    )
    parser.add_argument(
        "--stats_output",
        type=str,
        default=None,
        help="Path to save the core statistics CSV. If not set, print to console.",
    )
    parser.add_argument(
        "--ic_output",
        type=str,
        default=None,
        help="Path to save the factor IC results CSV. If not set, print to console.",
    )
    return parser.parse_args()


def core_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute core statistics for each column, returning a tidy DataFrame.
    """
    stats = []
    for col in df.columns:
        series = df[col]
        base = {"feature": col, "count": series.count(), "missing": series.isna().sum()}
        if pd.api.types.is_numeric_dtype(series):
            base.update(
                {
                    "sum": series.sum(),
                    "mean": series.mean(),
                    "median": series.median(),
                    "mode": series.mode().iloc[0] if not series.mode().empty else pd.NA,
                    "min": series.min(),
                    "q1": series.quantile(0.25),
                    "q3": series.quantile(0.75),
                    "max": series.max(),
                    "variance": series.var(),
                    "std": series.std(),
                    "cv": series.std() / series.mean() if series.mean() != 0 else pd.NA,
                    "skew": series.skew(),
                    "kurtosis": series.kurtosis(),
                }
            )
        stats.append(base)
    result = pd.DataFrame(stats).set_index("feature")
    return result


def corr_analysis(
    df: pd.DataFrame, factor_names: list, threshold: float, save_fig: str = None
):
    """
    Perform correlation analysis, reporting high-correlation pairs and plotting heatmap.
    """
    # Check for missing columns
    missing_cols = [col for col in factor_names if col not in df.columns]
    if missing_cols:
        sys.exit(f"Error: Missing columns in input: {missing_cols}")
    sub = df[factor_names].dropna()
    corr = sub.corr()

    # Report high correlations
    pairs = corr.where(lambda x: x.abs() >= threshold).stack().reset_index()
    pairs.columns = ["Feature 1", "Feature 2", "Correlation"]
    pairs = pairs[pairs["Feature 1"] < pairs["Feature 2"]]

    if not pairs.empty:
        print("\nHigh-Correlation Pairs (|r| >= {:.2f}):".format(threshold))
        print(pairs.to_string(index=False, float_format="{:0.4f}".format))
    else:
        print(f"\nNo feature pairs with |r| >= {threshold:.2f} found.")

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Correlation"},
    )
    plt.title("Factor Correlation Heatmap")
    plt.tight_layout()

    if save_fig:
        plt.savefig(save_fig)
        print(f"Heatmap saved to: {save_fig}")
    else:
        plt.show()


def ic_analysis(
    df: pd.DataFrame,
    factor_names: list,
    return_col: str,
    output: str = None,
    date_col: str = "Date",
):
    """
    Compute cross-sectional IC (Spearman) summary for each factor vs return.
    Requires a date column for grouping.
    Output: IC_mean, IC_std, IC_t, IC_IR
    """
    if return_col not in df.columns:
        sys.exit(f"Error: Return column '{return_col}' not found in input file.")
    if date_col not in df.columns:
        sys.exit(f"Error: Date column '{date_col}' not found in input file.")

    summary_list = []

    for fac in factor_names:
        if fac not in df.columns:
            continue

        # 每日截面计算IC
        daily_ic = (
            df[[date_col, fac, return_col]]
            .dropna()
            .groupby(date_col, group_keys=False)
            .apply(lambda x: x[fac].corr(x[return_col], method="spearman"))
        )

        # 统计指标
        ic_mean = daily_ic.mean()
        ic_std = daily_ic.std(ddof=1)
        ic_ir = ic_mean / ic_std if ic_std > 0 else pd.NA

        summary_list.append(
            {
                "factor": fac,
                "IC_mean": ic_mean,
                "IC_std": ic_std,
                "IC_IR": ic_ir,
            }
        )

    summary_df = pd.DataFrame(summary_list).set_index("factor")

    if output:
        summary_df.to_csv(output)
        print(f"IC summary saved to: {output}")
    else:
        print("\nFactor IC Summary (Spearman, cross-sectional):")
        print(summary_df.round(4).to_string())

    return summary_df


def main():
    args = parse_args()
    names = [n.strip() for n in args.factor_names.split(",")]
    df = pd.read_csv(args.factor_file)

    # Core statistics
    stats = core_statistics(df)
    if args.stats_output:
        stats.to_csv(args.stats_output)
        print(f"Core statistics saved to: {args.stats_output}")
    else:
        print("\nCore Statistics:")
        print(stats.round(4).to_string())

    # Correlation analysis
    corr_analysis(
        df=df, factor_names=names, threshold=args.threshold, save_fig=args.save_fig
    )

    # IC analysis
    ic_analysis(
        df=df, factor_names=names, return_col=args.return_col, output=args.ic_output
    )


if __name__ == "__main__":
    main()
