import pandas as pd
import numpy as np

from aiq.ops import Ref, Resi, Log, Rank, Sum, Corr, Std, IdxMax, IdxMin, Slope


def test_ref():
    input_series = pd.Series([10, 20, 30, 40])
    expected = pd.Series([np.nan, np.nan, 10.0, 20.0])
    output = Ref(input_series, 2)
    pd.testing.assert_series_equal(output, expected)


def test_resi():
    input_series = pd.Series([10, 20, 30, 40])
    expected = pd.Series([np.nan, 0.0, 0.0, 0.0])
    output = Resi(input_series, 2)
    pd.testing.assert_series_equal(output, expected)


def test_log():
    input_series = pd.Series([1, np.e, np.e**2])
    expected = pd.Series([0.0, 1.0, 2.0])
    output = Log(input_series)
    pd.testing.assert_series_equal(output, expected)


def test_rank():  # 修正后的测试
    input_series = pd.Series([10, 20, 30, 15])
    expected = pd.Series([1.0, 1.0, 1.0, 0.5])  # 修正期望值
    output = Rank(input_series, 2)
    pd.testing.assert_series_equal(output, expected)


def test_sum():
    input_series = pd.Series([1, 2, 3, 4])
    expected = pd.Series([1.0, 3.0, 5.0, 7.0])
    output = Sum(input_series, 2)
    pd.testing.assert_series_equal(output, expected)


def test_corr_perfect_linear():
    """
    对完全线性（且相同）的两条序列，任意窗口大小下，
    只要 window >= 2，相关系数恒为 1。
    """
    s1 = pd.Series([10, 20, 30, 40, 50])
    s2 = pd.Series([10, 20, 30, 40, 50])
    N = 2

    # 对 window=2，min_periods=1 时：
    # idx 0: 只有 1 个点 → NaN（PairRolling 默认）
    # idx >=1: 两点以上都刚好是完全线性 → corr = 1.0
    expected = pd.Series([np.nan, 1.0, 1.0, 1.0, 1.0])

    output = Corr(s1, s2, N)
    pd.testing.assert_series_equal(output, expected)


def test_corr_zero_variance():
    """
    如果任一条序列在窗口内标准差≈0，则强制置 NaN。
    这里用常数序列和任意序列做测试。
    """
    s1 = pd.Series([5, 5, 5, 5, 5])  # 零方差
    s2 = pd.Series([1, 2, 3, 4, 5])  # 正常变化
    N = 3

    # 任意窗口下，s1.std()==0 → 全部 NaN
    expected = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])

    output = Corr(s1, s2, N)
    pd.testing.assert_series_equal(output, expected)


def test_std_basic_window2():
    """
    简单线性序列，window=2 时的滚动 std（ddof=1）：
    std([1,2]) = sqrt(0.5) ≈ 0.70710678
    """
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    N = 2

    expected = pd.Series(
        [
            np.nan,
            np.sqrt(0.5),
            np.sqrt(0.5),
            np.sqrt(0.5),
        ],
        index=s.index,
    )

    output = Std(s, N)
    pd.testing.assert_series_equal(output, expected)


def test_std_constant_series():
    """
    常数序列，任何 window >=2 时方差为0，对应 std=0，
    且不足 window 的位置返回 NaN。
    """
    s = pd.Series([5.0, 5.0, 5.0, 5.0])
    N = 3

    expected = pd.Series(
        [
            np.nan,  # idx=0,1 不足 2 点
            0.0,
            0.0,  # idx=2 有 2 个点，全相同 → std=0
            0.0,  # idx=3 同上
        ],
        index=s.index,
    )

    output = Std(s, N)
    pd.testing.assert_series_equal(output, expected)


def test_std_matches_pandas():
    """
    随机数据下，Std 的结果应与 pandas.Series.rolling.std 完全一致
    （这里假设 Rolling 默认 min_periods=1，如需改则同步调整）。
    """
    rng = np.random.RandomState(0)
    s = pd.Series(rng.normal(size=50))
    N = 5

    output = Std(s, N)
    # pandas 默认 ddof=1，min_periods=1
    expected = s.rolling(window=N, min_periods=1).std()

    pd.testing.assert_series_equal(output, expected)


def test_idxmax_basic_window3():
    """
    基本功能：window=3 时，每个窗口中最大值的位置（1-based）。
    序列 [1,3,2,5,4]：
      idx0: only [1]         → argmax=0 +1 =1
      idx1: [1,3]            → argmax=1 +1 =2
      idx2: [1,3,2]          → argmax=1 +1 =2
      idx3: [3,2,5]          → argmax=2 +1 =3
      idx4: [2,5,4]          → argmax=1 +1 =2
    """
    s = pd.Series([1, 3, 2, 5, 4])
    expected = pd.Series([1.0, 2.0, 2.0, 3.0, 2.0], index=s.index)
    output = IdxMax(s, 3)
    pd.testing.assert_series_equal(output, expected)


def test_idxmax_ties_and_nans():
    """
    处理并列最大值和 NaN：
    - 并列最大时取首次出现的位置
    - NaN 被忽略，window 内若全 NaN 则返回 NaN
    """
    s = pd.Series([np.nan, 2, 2, np.nan, 1, 3, 3])
    N = 4
    # 计算每个位置的 window
    # idx0: [nan]          → all na → NaN
    # idx1: [nan,2]        → max=2 first at pos1 → 2
    # idx2: [nan,2,2]      → max=2 first at pos1 → 2
    # idx3: [nan,2,2,nan]  → same → 2
    # idx4: [2,2,nan,1]    → max=2 first at pos0 → 1
    # idx5: [2,nan,1,3]    → max=3 at pos3 → 4
    # idx6: [nan,1,3,3]    → max=3 first at pos2 → 3
    expected = pd.Series([np.nan, 2, 2, 2, 1, 4, 3], index=s.index)
    output = IdxMax(s, N)
    pd.testing.assert_series_equal(output, expected)


def test_idxmax_expanding():
    """
    扩展模式（N=0）：相当于 expanding，返回到当前为止的最大值位置。
    序列 [4,1,3,5]:
      idx0: [4]       → pos0+1 =1
      idx1: [4,1]     → max=4 at pos0 →1
      idx2: [4,1,3]   → max=4 at pos0 →1
      idx3: [4,1,3,5] → max=5 at pos3 →4
    """
    s = pd.Series([4, 1, 3, 5])
    expected = pd.Series([1.0, 1.0, 1.0, 4.0], index=s.index)
    output = IdxMax(s, 0)
    pd.testing.assert_series_equal(output, expected)


def test_idxmin_basic_window3():
    """
    基础用例：正常计算 min 的位置（1-based）
    s = [3, 2, 4, 1, 5]
    window=3:
    - idx=0: [3] → 1
    - idx=1: [3,2] → 2
    - idx=2: [3,2,4] → 2
    - idx=3: [2,4,1] → 3
    - idx=4: [4,1,5] → 2
    """
    s = pd.Series([3, 2, 4, 1, 5])
    expected = pd.Series([1.0, 2.0, 2.0, 3.0, 2.0], index=s.index)
    output = IdxMin(s, 3)
    pd.testing.assert_series_equal(output, expected)


def test_idxmin_ties_and_nans():
    """
    并列最小值 + NaN 情况
    - 并列最小取首次
    - 全 NaN 窗口 → NaN
    """
    s = pd.Series([np.nan, 2, 1, 1, np.nan, 0, 0])
    N = 3
    # 窗口移动过程大致如下：
    # idx0: [nan] → NaN
    # idx1: [nan,2] → 2 at pos1 → 2
    # idx2: [nan,2,1] → 1 at pos2 → 3
    # idx3: [2,1,1] → 1 first at pos1 → 2
    # idx4: [1,1,nan] → 1 first at pos0 → 1
    # idx5: [1,nan,0] → 0 at pos2 → 3
    # idx6: [nan,0,0] → 0 at pos1 → 2
    expected = pd.Series([np.nan, 2.0, 3.0, 2.0, 1.0, 3.0, 2.0], index=s.index)
    output = IdxMin(s, N)
    pd.testing.assert_series_equal(output, expected)


def test_idxmin_expanding():
    """
    expanding 模式：每步返回当前为止的最小值位置
    s = [4, 3, 5, 1]
    - idx0: [4] → 1
    - idx1: [4,3] → 2
    - idx2: [4,3,5] → 2
    - idx3: [4,3,5,1] → 4
    """
    s = pd.Series([4, 3, 5, 1])
    expected = pd.Series([1.0, 2.0, 2.0, 4.0], index=s.index)
    output = IdxMin(s, 0)
    pd.testing.assert_series_equal(output, expected)


def test_slope_rolling_perfect_linear():
    """
    完全线性增长，斜率应为常数。
    y = 3x + 1 → slope = 3
    """
    s = pd.Series([4, 7, 10, 13, 16])  # y = 3x + 1
    N = 3
    expected = pd.Series([np.nan, 3.0, 3.0, 3.0, 3.0])
    output = Slope(s, N)
    pd.testing.assert_series_equal(output, expected)


def test_slope_rolling_with_nan():
    """
    NaN 不应参与计算，导致有效点数不足时结果为 NaN
    """
    s = pd.Series([1.0, np.nan, 3.0, 6.0, 9.0])
    N = 3
    # [1, nan, 3] → 有效点 < 3 → NaN
    # [nan, 3, 6] → 有效点 < 3 → NaN
    # [3, 6, 9] → slope = 3
    expected = pd.Series([np.nan, np.nan, 1.0, 3.0, 3.0])
    output = Slope(s, N)
    pd.testing.assert_series_equal(output, expected)


def test_slope_expanding_mode():
    """
    expanding 模式，等价于从头开始做线性回归。
    s = [2, 4, 6, 8, 10] → 每个扩展窗口的斜率都应为 2
    """
    s = pd.Series([2, 4, 6, 8, 10])
    N = 0  # expanding 模式
    expected = pd.Series([np.nan, 2.0, 2.0, 2.0, 2.0])
    output = Slope(s, N)
    pd.testing.assert_series_equal(output, expected)


def test_slope_compare_with_manual():
    """
    与手动计算斜率结果比对，确保一致。
    """

    def linear_slope(y):
        N = len(y)
        x = np.arange(1, N + 1)
        x_mean = x.mean()
        y_mean = np.mean(y)
        num = ((x - x_mean) * (y - y_mean)).sum()
        denom = ((x - x_mean) ** 2).sum()
        return num / denom if denom != 0 else np.nan

    s = pd.Series([1, 2, 3, 2, 1, 0])
    N = 3
    expected = s.rolling(N, min_periods=1).apply(linear_slope, raw=True)
    output = Slope(s, N)
    pd.testing.assert_series_equal(output, expected)
