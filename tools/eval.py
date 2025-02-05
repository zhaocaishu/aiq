import os
import argparse
import pickle

from aiq.utils.config import config as cfg
from aiq.utils.module import init_instance_by_config
from aiq.utils.logging import get_logger
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

    # logger
    logger = get_logger("EVAL")

    logger.info(cfg)

    # data handler
    with open(os.path.join(args.save_dir, "data_handler.pkl"), "rb") as f:
        data_handler = pickle.load(f)

    # dataset
    val_dataset = init_instance_by_config(
        cfg.dataset,
        data_dir=args.data_dir,
        data_handler=data_handler,
        mode="valid",
    )
    logger.info("Loaded %d items to validation dataset" % len(val_dataset))

    # load model
    model = init_instance_by_config(
        cfg.model,
        feature_cols=val_dataset.feature_names,
        label_col=[val_dataset.label_name],
        logger=logger,
    )
    model.load(args.save_dir)

    # prediction
    pred_df = model.predict(val_dataset).data

    # evaluation
    evaluator = Evaluator()
    metrics = evaluator.evaluate(pred_df)
    logger.info("Evaluation metrics: %s" % str(metrics))


if __name__ == "__main__":
    main()
