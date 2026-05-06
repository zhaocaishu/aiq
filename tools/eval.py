import os
import argparse

from aiq.utils.config import config as cfg
from aiq.utils.module import init_instance_by_config
from aiq.utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model")

    parser.add_argument(
        "--cfg_file",
        type=str,
        required=True,
        help="Path to the configuration file for evaluation.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory path of evaluation data.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save evaluation results.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Data split for evaluation.",
    )
    parser.add_argument(
        "--model_names",
        type=str,
        nargs="+",
        default=["model.pth"],
        help="Model checkpoint names for ensemble, e.g. model1.pth model2.pth",
    )

    return parser.parse_args()


def load_models(cfg, val_dataset, save_dir: str, model_names, logger) -> list:
    models = []

    for model_name in model_names:
        logger.info(f"Loading model: {model_name}")
        model = init_instance_by_config(
            cfg.model,
            feature_names=val_dataset.feature_names,
            label_names=val_dataset.label_names,
            save_dir=save_dir,
            logger=logger,
        )
        model.load(model_name)
        models.append(model)

    logger.info("Total models loaded: %d", len(models))
    return models


def ensemble_predict(models, dataset, logger):
    """
    Run prediction for multiple models and average prediction columns.
    """
    pred_dfs = []

    for i, model in enumerate(models):
        logger.info(f"Running prediction for model {i + 1}/{len(models)}")
        pred_df = model.predict(dataset).reset_index()
        pred_dfs.append(pred_df)

    # 以第一个为基准
    base_df = pred_dfs[0].copy()
    pred_cols = [c for c in base_df.columns if c.startswith("PRED_")]

    logger.info("Ensembling prediction columns: %s", pred_cols)

    for col in pred_cols:
        base_df[col] = sum(df[col] for df in pred_dfs) / len(pred_dfs)

    return base_df


def main():
    args = parse_args()

    # Load config
    cfg.from_file(args.cfg_file)

    logger = get_logger("EVALUATION")
    logger.info("Starting evaluation with config:\n%s", cfg)

    data_handler = init_instance_by_config(cfg.data_handler, data_dir=args.data_dir)
    data_handler.load(os.path.join(args.save_dir, "data_handler.pkl"))
    eval_data = data_handler.setup_data(mode=args.split)
    logger.info("Data handler completed. Shape: %s", eval_data.shape)

    # Load dataset
    eval_dataset = init_instance_by_config(
        cfg.dataset,
        data=eval_data,
        data_dir=args.data_dir,
        feature_names=data_handler.feature_names,
        mode=args.split,
    )
    logger.info("Evaluation dataset loaded: %d samples", len(eval_dataset))

    # Load models
    models = load_models(
        cfg,
        eval_dataset,
        args.save_dir,
        args.model_names,
        logger,
    )

    # Prediction
    pred_df = ensemble_predict(models, eval_dataset, logger)
    logger.info("Ensemble prediction completed. Shape: %s", pred_df.shape)

    # Evaluation
    evaluator = init_instance_by_config(
        cfg.evaluator, data_dir=args.data_dir, logger=logger
    )
    metrics = evaluator.evaluate(pred_df)
    logger.info("Evaluation metrics:\n%s", metrics)


if __name__ == "__main__":
    main()
