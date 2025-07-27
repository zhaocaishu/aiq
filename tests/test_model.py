import torch

from aiq.utils.config import config as cfg
from aiq.utils.module import init_instance_by_config
from aiq.utils.logging import get_logger


if __name__ == "__main__":
    # config
    cfg.from_file("./configs/ppnet_model_reg.yaml")

    # logger
    logger = get_logger("MODEL")

    model = init_instance_by_config(
        cfg.model,
        save_dir="./checkpoints",
        logger=logger,
    )

    logger.info("Model initialized successfully")

    dummy_input = torch.zeros(100, 16, 206)
    output = model.model(dummy_input)
    logger.info(output)
