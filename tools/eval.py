import os
import argparse
import pickle

from aiq.dataset import inst_data_handler, Dataset
from aiq.models import XGBModel, LGBModel, DEnsembleModel, PatchTSTModel, NLinearModel
from aiq.utils.config import config as cfg


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

    # load data handler
    print(cfg.dataset.segments)
    with open(os.path.join(args.save_dir, "data_handler.pkl"), "rb") as f:
        data_handler = pickle.load(f)

    # dataset
    val_dataset = Dataset(
        args.data_dir,
        instruments=cfg.market,
        start_time=cfg.dataset.segments["valid"][0],
        end_time=cfg.dataset.segments["valid"][1],
        data_handler=data_handler,
        mode="valid",
    )
    print("Loaded %d items to test dataset" % len(val_dataset))

    # model
    if cfg.model.name == "XGB":
        model = XGBModel()
    elif cfg.model.name == "LGB":
        model = LGBModel()
    elif cfg.model.name == "DoubleEnsemble":
        model = DEnsembleModel()
    elif cfg.model.name == "PatchTST":
        model = PatchTSTModel(model_params=cfg.model.params)
    elif cfg.model.name == "NLinear":
        model = NLinearModel(model_params=cfg.model.params)
    model.load(args.save_dir)

    # evaluation
    result = model.eval(val_dataset)
    print("Evaluation metric result:", result)


if __name__ == "__main__":
    main()
