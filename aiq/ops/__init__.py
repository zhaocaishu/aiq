from .ops import (
    Ref,
    Mean,
    Std,
    Var,
    Skew,
    Kurt,
    Max,
    IdxMax,
    Min,
    IdxMin,
    Quantile,
    Med,
    Mad,
    Rank,
    Count,
    Delta,
    Slope,
    Rsquare,
    Resi,
    WMA,
    EMA,
    Greater,
    Less,
    Corr,
    Log,
    Abs,
    Sum,
    Cov,
    CSRank,
)

Greater: Greater = Greater()
Less: Less = Less()
Log: Log = Log()
Abs: Abs = Abs()
Ref: Ref = Ref()
Mean: Mean = Mean()
Sum: Sum = Sum()
Std: Std = Std()
Var: Var = Var()
Skew: Skew = Skew()
Kurt: Kurt = Kurt()
Max: Max = Max()
IdxMax: IdxMax = IdxMax()
Min: Min = Min()
IdxMin: IdxMin = IdxMin()
Quantile: Quantile = Quantile()
Med: Med = Med()
Mad: Mad = Mad()
Rank: Rank = Rank()
Count: Count = Count()
Delta: Delta = Delta()
Slope: Slope = Slope()
Rsquare: Rsquare = Rsquare()
Resi: Resi = Resi()
WMA: WMA = WMA()
EMA: EMA = EMA()
Corr: Corr = Corr()
Cov: Cov = Cov()
CSRank: CSRank = CSRank()

__all = [
    "Rolling",
    "Ref",
    "Mean",
    "Std",
    "Var",
    "Skew",
    "Kurt",
    "Max",
    "IdxMax",
    "Min",
    "IdxMin",
    "Quantile",
    "Med",
    "Mad",
    "Rank",
    "Count",
    "Delta",
    "Slope",
    "Rsquare",
    "Resi",
    "WMA",
    "EMA",
    "Greater",
    "Less",
    "Corr",
    "Log",
    "Abs",
    "Sum",
    "Cov",
    "CSRank",
]
