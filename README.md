# AIQ

## 简介

AIQ是基于AI算法的金融量化模型训练代码框架，集成了树模型、深度学习模型以及时序预测模型等。

## 安装

```
git clone https://github.com/zhaocaishu/aiq.git
cd aiq
pip3 install .
```

## 使用

在完成环境设置后，您可以按照以下说明开始使用 AIQ：

1. **准备输入数据**

    利用 https://github.com/zhaocaishu/aiq-datasets 项目里的脚本完成数据准备。

2. **训练**
    ```
    python3 tools/train.py \
        --cfg_file ./configs/xgboost_model_reg.yaml \
        --data_dir ./data \
        --seeds 1234 2024 3407 \
        --save_dir ./output
    ```

3. **评估**
    ```
    python3 tools/eval.py \
        --cfg_file ./configs/xgboost_model_reg.yaml \
        --data_dir ./data \
        --split test \
        --model_names model_seed_1234.pth  model_seed_2024.pth  model_seed_3407.pth \
        --save_dir ./output
    ```

    评估日志分为排序能力、TopK 选股表现、策略回测、十分组收益和月度 RankIC 五组结果。收益、命中率、覆盖率和换手率以百分比显示，RankIC、RankICIR 和 Sharpe 保留四位小数，无法计算的值显示为 `--`。主要统计指标如下：

    - `RankIC` / `RankICIR`：每日 Spearman 相关系数的均值及其均值/标准差。
    - `HitRate@TopK` / `WorstHit@TopK`：预测 TopK 中命中真实最好/最差 K 只股票的比例。
    - `TopReturn@TopK`：预测 TopK 的平均标签收益；`ExcessReturn@TopK` 减去同期指数基准收益，`PoolExcess@TopK` 减去当日有有效预测的股票池等权收益。
    - `LabelCoverage@TopK`：已选 TopK 中有效标签的比例。先按预测选股，再统计可观测标签，不因未来标签缺失而补选；收益、命中率和胜率只统计已选股票中标签可观测的部分。
    - 十分组收益：`Q1` 为预测最低组，`Q10` 为最高组，同分股票保持同组；`Q10-Q1` 为每日两组收益差的均值。
    - 月度 IC：按月汇总每日 `RankIC`，`Days` 为有效 RankIC 的日期数；十分组表中的 `Days` 为对应组收益有效的日期数。

    统计范围为当日股票池内有有效预测的股票，各项先按日计算，再对有效日期等权平均。标签应为未经标准化或裁剪的复权远期收益，标签收益诊断不扣费、不年化。真实尾部排名使用标签可观测的候选股票；候选不足 `2K` 或标签全相同时，命中率和误选率为 `NaN`。

    评估默认增加风险诊断，可在 `evaluator.kwargs` 设置 `risk_analysis: false` 关闭：

    - **预测风险暴露**：分数与规模、20日收益波动、20日换手、60日市场 beta 的 Spearman 相关，以及 TopK 相对股票池的暴露差；行业暴露为权重差。
    - **同风险组内评估**：每天按波动、规模分别分为五组，并按行业分组，报告组内 RankIC、高分20%相对组均值收益及高低分收益差。风险或分数相同的样本不强行拆分；分位边界重复时合并，分组不足30只不计算表现。
    - **TopK 收益归因**：每日用行业、规模、波动、换手及 beta 回归原始远期收益，按年汇总共同样本上的池内超额、各风险贡献及残差贡献。至少需要100只且不少于2K只完整样本，回归满秩且有剩余自由度，否则报告跳过原因。

    风险指标仅使用信号日收盘及之前的数据，自动多读180个自然日的历史。股票和指数按交易日历对齐，缺失报价不填充、不跨缺失日计算日收益，完整20/60日窗口不足时保留缺失。连续风险先按当日预测池1%/99%分位数去极值，再标准化；行业不标准化。缺失行业或风险字段会反映在覆盖率中。

    TopK 在过滤风险和标签前选定，缺失不补选。归因使用风险及标签完整的共同样本，其 PoolExcess 可能不同于原全池指标。因子贡献是事后统计解释，不能用于当时的交易信号；残差不等于纯 alpha。报告不解释实际轮换持仓的回测收益，也不对重叠标签计算独立样本显著性。

    `Evaluator.evaluate()` 后，`evaluator.risk_results` 保存 `exposures`、`groups`、`attribution` 三张每日数值表；可调用 `evaluator.risk_evaluator.summarize(evaluator.risk_results, period="M")` 或 `period="Y"` 查看月度、年度结果。`Days` 为尝试评估的日期数，各指标的 `*Days` 为实际有效日期数。也可以直接使用 `RiskEvaluator.prepare_features()` 和 `evaluate()` 分析已有预测，无需重新加载模型。
