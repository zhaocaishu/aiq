# AIQ

AIQ 是面向 A 股选股研究的量化模型框架，支持 PPNet、LightGBM、XGBoost 等模型，涵盖特征处理、模型训练、预测、选股评测与策略回测。

## 安装与数据准备

```bash
git clone https://github.com/zhaocaishu/aiq.git
cd aiq
pip3 install .
```

使用 [aiq-datasets](https://github.com/zhaocaishu/aiq-datasets) 中的脚本准备输入数据。下文命令均在项目根目录执行，假设数据位于 `./data`，结果保存到 `./output`。

## 运行入口

| 入口 | 用途 | 主要输出 |
|:-----|:-----|:---------|
| `tools/train.py` | 普通训练，使用训练集和验证集 | 模型文件、预处理参数 |
| `tools/eval.py` | 加载模型预测并评测，或单独评测已有预测文件 | 预测文件、评测报告；单独评测时仅输出报告 |
| `tools/rolling_train.py` | 按月滚动训练和预测 | 各月模型、预处理参数、预测文件，以及合并预测文件 |

三个入口共用 `aiq/utils/workflow.py` 中的数据集构建、训练、预测及评测函数。

| 配置文件 | 适用流程 |
|:---------|:---------|
| `configs/ppnet_model_reg.yaml` | PPNet 普通训练、预测与评测 |
| `configs/ppnet_model_rolling.yaml` | PPNet 滚动训练、预测及合并结果评测 |
| `configs/lightgbm_model_reg.yaml` | LightGBM 普通训练、预测与评测 |
| `configs/xgboost_model_reg.yaml` | XGBoost 普通训练、预测与评测 |

## 普通训练与评测

### 1. 训练模型

```bash
python3 tools/train.py \
    --cfg_file ./configs/ppnet_model_reg.yaml \
    --data_dir ./data \
    --seeds 1234 2024 3407 \
    --data_seed 1234 \
    --save_dir ./output/ppnet
```

配置需要包含 `dataset.kwargs.segments.train` 和 `valid`。训练完成后，输出目录保存：

- `data_handler.pkl`：已拟合的预处理参数，供后续预测复用。
- `model_seed_<seed>.pth`：每个训练随机种子对应的模型文件。

`--seeds` 控制模型训练随机种子，`--data_seed` 控制数据准备的随机性。训练结束后，使用下一步命令生成预测和评测报告。

### 2. 预测并评测

```bash
python3 tools/eval.py \
    --cfg_file ./configs/ppnet_model_reg.yaml \
    --data_dir ./data \
    --model_dir ./output/ppnet \
    --model_names model_seed_1234.pth model_seed_2024.pth model_seed_3407.pth \
    --data_seed 1234 \
    --split test \
    --save_dir ./output/ppnet/test
```

预测时沿用训练配置和 `--data_seed`，加载模型及 `data_handler.pkl`。多个模型的预测按股票和日期对齐后取平均，输出原始收益标签及预测分数。

输出目录包含 `predictions.csv` 和 `metrics.txt`。评测读取刚保存的预测文件，与单独评测已有文件使用同一流程。

`--split` 指定预测和评测的数据段，日期取自 `dataset.kwargs.segments`。即使选择 `train` 或 `valid`，预测也复用已拟合的预处理参数，并保留原始收益标签。

### 3. 单独评测已有预测

已有 `predictions.csv` 时，可直接重新评测：

```bash
python3 tools/eval.py \
    --cfg_file ./configs/ppnet_model_reg.yaml \
    --data_dir ./data \
    --pred_file ./output/ppnet/test/predictions.csv \
    --split test \
    --save_dir ./output/ppnet/retest
```

该方式只输出 `metrics.txt`，不加载模型。评测仍需要 `--data_dir` 中的行情、股票池和基准数据；`--pred_file` 不能与 `--model_dir` 或 `--model_names` 同时使用。

## 滚动训练与评测

### 1. 按月训练和预测

```bash
python3 tools/rolling_train.py \
    --cfg_file ./configs/ppnet_model_rolling.yaml \
    --data_dir ./data \
    --seeds 1234 2024 3407 \
    --data_seed 1234 \
    --save_dir ./output/rolling_2023
```

当前滚动配置预测 2023 年，训练窗口约 6 年，逐月平移；不设置验证集，使用固定训练轮数，并保留 5 个交易日的标签隔离。该配置由 `rolling_train.py` 运行。

每次运行只提取一次共享原始因子并在内存中复用，各月独立拟合预处理参数、训练模型并预测。输出结构如下：

```text
output/rolling_2023/
├── predictions.csv          # 合并整个预测区间的结果
├── 2023-01/
│   ├── train_config.yaml
│   ├── predict_config.yaml
│   ├── data_handler.pkl
│   ├── model_seed_1234.pth   # 其他种子的模型也保存在此目录
│   └── predictions.csv      # 当月预测
├── 2023-02/
│   └── ...
└── ...
```

合并预测时会检查股票与日期组合的唯一性，以及交易日覆盖情况。滚动入口完成预测后结束，评测使用下一步命令。

### 2. 统一评测滚动结果

```bash
python3 tools/eval.py \
    --cfg_file ./configs/ppnet_model_rolling.yaml \
    --data_dir ./data \
    --pred_file ./output/rolling_2023/predictions.csv \
    --split test \
    --save_dir ./output/rolling_2023/test
```

使用根输出目录中的合并预测文件，统一计算全年选股指标和连续回测结果，生成 `metrics.txt`。各月模型和预处理参数位于月度子目录；全年评测通过 `--pred_file` 读取合并结果。

## 参数与预测文件

### 常用参数

| 参数 | 说明 |
|:-----|:-----|
| `--cfg_file` | 配置文件路径，必填 |
| `--data_dir` | 输入数据目录，必填；单独评测也需要 |
| `--save_dir` | 当前任务的输出目录，必填 |
| `--seeds` | 普通训练和滚动训练的模型随机种子，默认 `1234`，支持多个 |
| `--data_seed` | 数据准备随机种子，默认 `1234`；预测时与训练保持一致 |
| `--model_dir` | 预测时加载模型和预处理参数的目录，默认使用 `--save_dir` |
| `--model_names` | 参与预测的模型文件名，默认仅 `model_seed_1234.pth`，不会自动加载目录内所有模型 |
| `--split` | 预测和评测的数据段，可选 `train`、`valid`、`test`，默认 `test` |
| `--pred_file` | 已有预测 CSV 的路径；提供后直接评测，跳过模型预测 |

### 预测文件格式

当前 PPNet 的 `predictions.csv` 包含以下列，其他配置的标签名以 `dataset.kwargs.label_names` 为准：

| 列名 | 内容 |
|:-----|:-----|
| `Instrument` | 股票代码，以字符串读取，保留前导零 |
| `Date` | 信号日期，保存格式为 `YYYY-MM-DD` |
| `RET_5D` | 原始复权远期收益标签，未经标准化或裁剪 |
| `PRED_RET_5D` | 模型预测分数，即 `PRED_<标签名>` |

股票与日期组合必须唯一，预测分数必须为有限数值。标签允许缺失，但不能为无穷值；缺失标签不会被填充或用于补选股票。

文件日期必须落在 `--split` 对应的区间内，超出区间会报错，不会自动截取。离线评测期末收益还需要相应的后续行情，否则相关标签可能缺失。

## 评测说明

报告包含排序能力、TopK 选股表现、策略回测、十分组收益、月度 RankIC，以及默认开启的风险诊断。`K` 由 `evaluator.kwargs.top_k` 指定。

收益、命中率、覆盖率和换手率以百分比显示；RankIC、RankICIR 和 Sharpe 保留四位小数；无法计算的值显示为 `--`。

### 排序与选股指标

这些指标在当日股票池内有有效预测的股票上计算，先逐日统计，再对有效日期等权平均。标签收益指标不扣费、不年化。

| 指标 | 含义 |
|:-----|:-----|
| `RankIC` / `RankICIR` | 每日 Spearman 相关系数的均值，以及均值除以标准差；RankICIR 未年化 |
| `HitRate` / `WorstHit` | 预测 TopK 中，标签可观测股票命中真实最好 / 最差 K 只股票的比例 |
| `WinRate` | 预测 TopK 中，标签收益高于同期指数基准收益的比例；仅统计股票与基准收益均可观测的样本 |
| `TopReturn` | 预测 TopK 的平均标签收益 |
| `ExcessReturn` | TopK 标签收益相对同期指数基准的超额收益 |
| `PoolExcess` | TopK 标签收益相对当日有效预测池等权收益的超额收益 |
| `LabelCoverage` | 已选 TopK 中标签可观测的比例 |
| `Q1` ～ `Q10` | 从最低预测分到最高预测分的十分组收益；同分股票保持同组 |
| `Q10-Q1` | 每日最高分组与最低分组收益差的均值 |
| 月度 `RankIC` | 按月汇总每日 RankIC，`Days` 为有效日期数 |

TopK 先按预测分数选定，再统计可观测标签，缺失不补选。真实尾部排名使用标签可观测的候选股票；候选不足 `2K` 或标签全相同时，`HitRate` 和 `WorstHit` 无法计算。十分组表中的 `Days` 为对应组收益有效的日期数。

### 策略回测指标

策略回测基于实际模拟持仓和交易费用计算，与上面的标签收益诊断分开报告。

| 指标 | 含义 |
|:-----|:-----|
| `ARR` | 策略年化收益率 |
| `Sharpe` | 基于每日策略收益的年化夏普比率 |
| `MaxDrawdown` | 最大回撤 |
| `AnnVol` | 年化波动率 |
| `AvgTurnover` | 平均单边换手率 |

### 风险诊断

默认开启，可在 `evaluator.kwargs` 中设置 `risk_analysis: false` 关闭。

| 诊断 | 内容 |
|:-----|:-----|
| 预测风险暴露 | 预测分数与规模、20 日波动、20 日换手、60 日市场 beta 的 Spearman 相关，以及 TopK 相对股票池的暴露差；行业暴露为权重差 |
| 同风险组内评估 | 每日按波动、规模分别分为五组，并按行业分组，统计组内 RankIC、高分 20% 相对组均值收益及高低分收益差 |
| TopK 收益归因 | 每日用行业、规模、波动、换手及 beta 回归原始远期收益，按年汇总共同样本上的池内超额、风险贡献和残差贡献 |

**数据与样本口径：**

- 风险指标仅使用信号日收盘及之前的数据，自动多读 180 个自然日的历史。股票和指数按交易日历对齐，缺失报价不填充、不跨缺失日计算日收益，完整 20 / 60 日窗口不足时保留缺失。
- 连续风险按当日预测池的 1% / 99% 分位数去极值后标准化；行业不标准化。缺失行业或风险字段会反映在覆盖率中。
- 同风险组内不强行拆分相同风险值或相同分数的样本；分位边界重复时合并，分组不足 30 只时不计算表现。
- 归因至少需要 100 只且不少于 `2K` 只完整样本，回归须满秩且有剩余自由度，否则报告跳过原因。

**结果解读：** TopK 在过滤风险和标签前选定，缺失不补选。归因基于风险及标签完整的共同样本，其 `PoolExcess` 可能不同于全池指标。因子贡献是事后统计解释，残差不等于纯 alpha；该报告不解释实际轮换持仓的回测收益，也不对重叠标签计算独立样本显著性。

**程序调用：** `Evaluator.evaluate()` 后，`evaluator.risk_results` 保存 `exposures`、`groups`、`attribution` 三张每日数值表。调用 `evaluator.risk_evaluator.summarize(evaluator.risk_results, period="M")` 或 `period="Y"` 可查看月度、年度结果。风险汇总中的 `Days` 为尝试评估的日期数，各指标的 `*Days` 为实际有效日期数。也可直接使用 `RiskEvaluator.prepare_features()` 和 `evaluate()` 分析已有预测。
