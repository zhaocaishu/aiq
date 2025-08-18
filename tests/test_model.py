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

    dummy_stock_features = torch.zeros(100, 16, 144).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    dummy_market_features = torch.zeros(100, 16, 63).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    dummy_industries = (
        torch.zeros(100).to("cuda" if torch.cuda.is_available() else "cpu").long()
    )
    output = model.model(dummy_stock_features, dummy_market_features, dummy_industries)
    logger.info(output)
