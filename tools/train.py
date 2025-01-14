import os
import argparse
import pickle

from aiq.dataset import inst_data_handler, Dataset
from aiq.models import XGBModel, LGBModel, DEnsembleModel
from aiq.utils.config import config as cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--cfg_file", type=str, default=None, help="specify the config for training"
    )
    parser.add_argument("--data_dir", type=str, help="the data directory")
    parser.add_argument("--save_dir", type=str, help="the saved directory")

    args = parser.parse_args()

    return args


def main():
    # parse args
    args = parse_args()

    # config
    cfg.from_file(args.cfg_file)
    print(cfg)

    # data handler
    data_handler_config = cfg["data_handler"]
    data_handler = inst_data_handler(data_handler_config)

    # dataset
    train_dataset = Dataset(
        args.data_dir,
        instruments=cfg.dataset.market,
        start_time=cfg.dataset.segments["train"][0],
        end_time=cfg.dataset.segments["train"][1],
        data_handler=data_handler,
        mode="train",
    )

    val_dataset = Dataset(
        args.data_dir,
        instruments=cfg.dataset.market,
        start_time=cfg.dataset.segments["valid"][0],
        end_time=cfg.dataset.segments["valid"][1],
        data_handler=data_handler,
        mode="valid",
    )

    print(
        "Loaded %d items to train dataset, %d items to validation dataset"
        % (len(train_dataset), len(val_dataset))
    )

    # train model
    if cfg.model.name == "XGB":
        model = XGBModel(
            feature_cols=train_dataset.feature_names,
            label_col=train_dataset.label_name,
            model_params=cfg.model.params,
        )
    elif cfg.model.name == "LGB":
        model = LGBModel(
            feature_cols=train_dataset.feature_names,
            label_col=train_dataset.label_name,
            model_params=dict(cfg.model.params),
        )
    elif cfg.model.name == "DoubleEnsemble":
        model = DEnsembleModel(
            feature_cols=train_dataset.feature_names,
            label_col=[train_dataset.label_name],
            **dict(cfg.model.params),
        )
    else:
        raise ValueError(f"Unsupported model name: {cfg.model.name}")

    model.fit(train_dataset=train_dataset, val_dataset=val_dataset)

    # save data hanlder and model
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    with open(os.path.join(args.save_dir, "data_handler.pkl"), "wb") as f:
        pickle.dump(data_handler, f)

    model.save(model_dir=args.save_dir)

    print("Model training has been finished successfully!")


if __name__ == "__main__":
    main()
