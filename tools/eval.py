import os
import argparse
import pickle

from aiq.utils.config import config as cfg
from aiq.dataset import Dataset
from aiq.models import XGBModel, LGBModel, DEnsembleModel, PatchTSTModel, NLinearModel
from aiq.evaluation import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument(
        "--cfg_file", type=str, default=None, help="specify the config for evaluation"
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
    with open(os.path.join(args.save_dir, "data_handler.pkl"), "rb") as f:
        data_handler = pickle.load(f)

    # dataset
    val_dataset = Dataset(
        args.data_dir,
        instruments=cfg.dataset.market,
        start_time=cfg.dataset.segments["valid"][0],
        end_time=cfg.dataset.segments["valid"][1],
        data_handler=data_handler,
        mode="valid",
    )
    print("Loaded %d items to validation dataset" % len(val_dataset))

    # load model
    if cfg.model.name == "XGB":
        model = XGBModel()
    elif cfg.model.name == "LGB":
        model = LGBModel()
    elif cfg.model.name == "DoubleEnsemble":
        model = DEnsembleModel()
    else:
        raise ValueError(f"Unsupported model name: {cfg.model.name}")
    model.load(args.save_dir)

    # prediction
    pred_df = model.predict(val_dataset).to_dataframe()

    # evaluation
    evaluator = Evaluator()
    results = evaluator.evaluate(pred_df)
    print("Evaluation result:", results)


if __name__ == "__main__":
    main()
