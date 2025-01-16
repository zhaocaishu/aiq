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

    利用https://github.com/zhaocaishu/aiq-datasets项目里的脚本完成数据准备。

2. **训练**
    ```
    python3 tools/train.py \
        --cfg_file ./configs/xgboost_model_reg.yaml \
        --data_dir ./data \
        --save_dir ./output
    ```

3. **评估**
    ```
    python3 tools/eval.py \
        --cfg_file ./configs/xgboost_model_reg.yaml \
        --data_dir ./data \
        --save_dir ./output
    ```