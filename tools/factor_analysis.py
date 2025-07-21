import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model")

    parser.add_argument(
        "--factor_file",
        type=str,
        required=True,
        help="Path to the factor file for analysis.",
    )

    parser.add_argument(
        "--factor_names",
        type=str,
        required=True,
        help="Comma-separated list of factor names, e.g., 'factor1,factor2,factor3'.",
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

    return parser.parse_args()


def compute_core_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    返回每个字段的核心统计信息，包括：
      - 样本量（count）
      - 缺失值数量（missing）
      - 总和（sum）
      - 均值（mean）
      - 中位数（median）
      - 众数（mode）
      - 最小值（min）
      - Q1（25% 分位数）
      - Q3（75% 分位数）
      - 最大值（max）
      - 方差（variance）
      - 标准差（std）
      - 变异系数（cv）
      - 偏度（skew）
      - 峰度（kurtosis）
    """
    stats = {}
    for col in df.columns:
        series = df[col]
        # 基本统计
        count = series.count()
        missing = series.isna().sum()
        # 仅对数值型计算
        if pd.api.types.is_numeric_dtype(series):
            s_sum = series.sum()
            mean = series.mean()
            median = series.median()
            mode = series.mode().iloc[0] if not series.mode().empty else None
            minimum = series.min()
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            maximum = series.max()
            variance = series.var()
            std = series.std()
            cv = std / mean if mean else None
            skew = series.skew()
            kurt = series.kurtosis()
        else:
            s_sum = mean = median = mode = minimum = q1 = q3 = maximum = variance = std = cv = skew = kurt = None

        stats[col] = {
            'count': count,
            'missing': missing,
            'sum': s_sum,
            'mean': mean,
            'median': median,
            'mode': mode,
            'min': minimum,
            'q1': q1,
            'q3': q3,
            'max': maximum,
            'variance': variance,
            'std': std,
            'cv': cv,
            'skew': skew,
            'kurtosis': kurt
        }

    return pd.DataFrame(stats).T


def corr_analysis(factor_df, factor_names, threshold=0.9, save_fig=None):
    # 检查列是否存在
    missing_cols = [col for col in factor_names if col not in factor_df.columns]
    if missing_cols:
        print(
            f"Error: The following columns are missing in the input file: {missing_cols}"
        )
        sys.exit(1)

    factor_df = factor_df[factor_names]
    correlation_matrix = factor_df.corr()

    # 找出高相关性因子对
    high_corr_pairs = []
    for i in range(len(factor_names)):
        for j in range(i + 1, len(factor_names)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) >= threshold:
                high_corr_pairs.append((factor_names[i], factor_names[j], corr_value))

    # 打印高相关因子对
    if high_corr_pairs:
        print("高相关性因子对（绝对值大于等于阈值）:")
        for f1, f2, corr in high_corr_pairs:
            print(f"{f1} 和 {f2} 的相关性为 {corr:.4f}")
    else:
        print("没有发现高相关性因子对。")

    # 可视化热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Factor Correlation Heatmap")

    if save_fig:
        plt.savefig(save_fig)
        print(f"图像已保存到 {save_fig}")
    else:
        plt.show()


def main():
    args = parse_args()

    factor_names = [name.strip() for name in args.factor_names.split(",")]
    factor_df = pd.read_csv(args.factor_file)

    # core statistics
    core_stats = compute_core_statistics(factor_df)
    print(core_stats)

    # correlation analysis
    corr_analysis(
        factor_df=factor_df,
        factor_names=factor_names,
        threshold=args.threshold,
        save_fig=args.save_fig,
    )


if __name__ == "__main__":
    main()
