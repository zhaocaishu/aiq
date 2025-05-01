import os
import argparse
import pickle

from aiq.utils.config import config as cfg
from aiq.utils.module import init_instance_by_config
from aiq.utils.logging import get_logger
from aiq.evaluation import Evaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model")

    parser.add_argument(
        "--cfg_file",
        type=str,
        default=None,
        help="Path to the configuration file for evaluation.",
    )
    parser.add_argument(
        "--eval_pred_col",
        type=str,
        default="PRED",
        help="Column name representing model predictions.",
    )
    parser.add_argument(
        "--eval_label_col",
        type=str,
        default="LABEL",
        help="Column name representing true labels.",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory path of evaluation data."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save evaluation results.",
    )

    return parser.parse_args()


def load_data_handler(save_dir: str, logger) -> object:
    handler_path = os.path.join(save_dir, "data_handler.pkl")
    if not os.path.exists(handler_path):
        logger.error(f"Data handler file not found: {handler_path}")
        raise FileNotFoundError(f"Missing file: {handler_path}")

    with open(handler_path, "rb") as f:
        logger.info(f"Loading data handler from {handler_path}")
        return pickle.load(f)


def load_model(cfg, val_dataset, save_dir: str, logger) -> object:
    logger.info("Initializing model...")
    model = init_instance_by_config(
        cfg.model,
        feature_cols=val_dataset.feature_names,
        label_cols=val_dataset.label_names,
        save_dir=save_dir,
        logger=logger,
    )
    model.load()
    logger.info("Model loaded successfully.")
    return model


def main():
    args = parse_args()

    # Load config
    if args.cfg_file:
        cfg.from_file(args.cfg_file)

    logger = get_logger("EVALUATION")
    logger.info("Starting evaluation with config:\n%s", cfg)

    data_handler = load_data_handler(args.save_dir, logger)

    # Load dataset
    val_dataset = init_instance_by_config(
        cfg.dataset,
        data_dir=args.data_dir,
        data_handler=data_handler,
        mode="valid",
    )
    logger.info("Validation dataset loaded: %d samples", len(val_dataset))

    # Load and predict
    model = load_model(cfg, val_dataset, args.save_dir, logger)
    pred_df = model.predict(val_dataset).data.reset_index()
    logger.info("Prediction completed. Shape: %s", pred_df.shape)

    # Evaluate
    evaluator = Evaluator(pred_col=args.eval_pred_col, label_col=args.eval_label_col)
    metrics = evaluator.evaluate(pred_df)
    logger.info("Evaluation metrics:\n%s", metrics)


if __name__ == "__main__":
    main()
