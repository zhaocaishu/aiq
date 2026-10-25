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

    评估日志包含汇总指标、十分组收益和月度 IC 三张表。主要统计指标如下：

    - `RankIC` / `RankICIR`：每日 Spearman 相关系数的均值及其均值/标准差。
    - `HitRate@TopK` / `WorstHit@TopK`：预测 TopK 中命中真实最好/最差 K 只股票的比例。
    - `TopReturn@TopK`：预测 TopK 的平均标签收益；`ExcessReturn@TopK` 减去同期指数基准收益，`PoolExcess@TopK` 减去当日有有效预测的股票池等权收益。
    - `LabelCoverage@TopK`：已选 TopK 中有效标签的比例。先按预测选股，再统计可观测标签，不因未来标签缺失而补选；收益、命中率和胜率只统计已选股票中标签可观测的部分。
    - 十分组收益：`Q1` 为预测最低组，`Q10` 为最高组，同分股票保持同组；`Q10-Q1` 为每日两组收益差的均值。
    - 月度 IC：按月汇总每日 `RankIC`，`Days` 为有效 RankIC 的日期数；十分组表中的 `Days` 为对应组收益有效的日期数。

    统计范围为当日股票池内有有效预测的股票，各项先按日计算，再对有效日期等权平均。标签应为未经标准化或裁剪的复权远期收益，标签收益诊断不扣费、不年化。真实尾部排名使用标签可观测的候选股票；候选不足 `2K` 或标签全相同时，命中率和误选率为 `NaN`。
