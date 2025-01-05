from aiq.dataset import inst_data_handler, Dataset
from aiq.models import LGBModel
from aiq.utils.config import config as cfg

if __name__ == "__main__":
    cfg.from_file("./configs/lightgbm_model_reg.yaml")

    # setup data handler
    data_handler_config = cfg["data_handler"]
    data_handler = inst_data_handler(data_handler_config)

    # datasets
    train_dataset = Dataset(
        "./data",
        instruments=["002750.SZ", "002811.SZ", "600490.SH"],
        start_time="2024-01-01",
        end_time="2024-04-30",
        data_handler=data_handler,
        training=True,
    )

    val_dataset = Dataset(
        "./data",
        instruments=["002750.SZ", "002811.SZ", "600490.SH"],
        start_time="2024-05-01",
        end_time="2024-09-30",
        data_handler=data_handler,
        training=True,
    )

    model_params = {
        "objective": "mse",
        "learning_rate": 0.2,
        "colsample_bytree": 0.8879,
        "max_depth": 8,
        "num_leaves": 210,
        "subsample": 0.8789,
        "lambda_l1": 205.6999,
        "lambda_l2": 580.9768,
        "metric": "rmse",
        "nthread": 4,
    }

    print(train_dataset.feature_names, train_dataset.label_name)

    # train stage
    model = LGBModel(
        feature_cols=train_dataset.feature_names,
        label_col=[train_dataset.label_name],
        model_params=model_params,
    )
    model.fit(train_dataset=train_dataset, val_dataset=val_dataset)
    model.save(model_dir="./temp")

    # predict stage
    model_eval = LGBModel()
    model_eval.load(model_dir="./temp")
    model_eval.predict(dataset=val_dataset)
