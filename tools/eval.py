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
        "--cfg_file",
        type=str,
        default=None,
        help="Specify the configuration file path for evaluation. If not provided, default settings will be used.",
    )

    parser.add_argument(
        "--eval_pred_col",
        type=str,
        default="PRED",
        help="Column name in the dataset representing the model predictions during evaluation.",
    )

    parser.add_argument(
        "--eval_label_col",
        type=str,
        default="LABEL",
        help="Column name in the dataset representing the true labels during evaluation.",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory path where the evaluation data is stored.",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory path where the evaluation results will be saved.",
    )

    args = parser.parse_args()

    return args


def main():
    # parse args
    args = parse_args()

    # config
    cfg.from_file(args.cfg_file)

    # logger
    logger = get_logger("EVALUATION")

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
        label_cols=[val_dataset.label_names],
        logger=logger,
    )
    model.load(args.save_dir)

    # prediction
    pred_df = model.predict(val_dataset).data

    # evaluation
    evaluator = Evaluator(pred_col=args.eval_pred_col, label_col=args.eval_label_col)
    metrics = evaluator.evaluate(pred_df)
    logger.info("Evaluation metrics: %s" % str(metrics))


if __name__ == "__main__":
    main()
