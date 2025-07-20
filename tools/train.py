import os
import argparse
import random
from typing import List, Any

import torch
import numpy as np

from aiq.utils.config import config as cfg
from aiq.utils.module import init_instance_by_config
from aiq.utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--cfg_file", type=str, required=True, help="Path to training config file"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory containing training data"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save model and data handler",
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="A seed for reproducible training."
    )
    return parser.parse_args()


def setup_logger(name: str = "TRAINING") -> Any:
    return get_logger(name)


def set_random_seed(seed):
    random.seed(seed)  # Python 随机种子
    np.random.seed(seed)  # NumPy 随机种子
    torch.manual_seed(seed)  # PyTorch CPU 随机种子
    torch.cuda.manual_seed(seed)  # PyTorch 当前 GPU 随机种子
    torch.cuda.manual_seed_all(seed)  # 所有 GPU 随机种子（多GPU训练）

    torch.backends.cudnn.deterministic = True  # 保证每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # 禁止 cuDNN 自动寻找最优算法（为了确定性）


def load_datasets(data: str, data_dir: str, feature_names: List[str]) -> tuple:
    # train dataset
    train_dataset = init_instance_by_config(
        cfg.dataset,
        data=data,
        data_dir=data_dir,
        feature_names=feature_names,
        mode="train",
    )

    val_dataset = init_instance_by_config(
        cfg.dataset,
        data=data,
        data_dir=data_dir,
        feature_names=feature_names,
        mode="valid",
    )
    return train_dataset, val_dataset


def train_and_save_model(
    train_dataset: Any, val_dataset: Any, save_dir: str, logger: Any
) -> None:
    model = init_instance_by_config(
        cfg.model,
        feature_names=train_dataset.feature_names,
        label_names=train_dataset.label_names,
        save_dir=save_dir,
        logger=logger,
    )
    model.fit(train_dataset=train_dataset, val_dataset=val_dataset)
    model.save()


def main():
    args = parse_args()

    # Load config
    cfg.from_file(args.cfg_file)

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    logger = get_logger("TRAINING")
    logger.info("Configuration loaded:\n%s", cfg)

    os.makedirs(args.save_dir, exist_ok=True)

    data_handler = init_instance_by_config(cfg.data_handler, data_dir=args.data_dir)
    data = data_handler.setup_data()
    logger.info("Data handler completed. Shape: %s", data.shape)
    data_handler.save(os.path.join(args.save_dir, "data_handler.pkl"))

    train_dataset, val_dataset = load_datasets(
        data, args.data_dir, data_handler.feature_names
    )
    logger.info(
        "Loaded %d training and %d validation samples.",
        len(train_dataset),
        len(val_dataset),
    )

    train_and_save_model(train_dataset, val_dataset, args.save_dir, logger)

    logger.info("Model training completed successfully!")


if __name__ == "__main__":
    main()
