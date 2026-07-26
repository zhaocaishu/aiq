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

    dummy_industry_ids = (
        torch.zeros(100, 2).to("cuda" if torch.cuda.is_available() else "cpu").long()
    )
    dummy_stock_ts_features = torch.zeros(100, 8, 26).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    dummy_stock_intraday_ts_features = torch.zeros(100, 384, 6).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    dummy_stock_cs_features = torch.zeros(100, 125).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    dummy_stock_fund_features = torch.zeros(100, 4).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    dummy_market_features = torch.zeros(100, 20).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    output = model.model(
        dummy_stock_ts_features,
        dummy_stock_cs_features,
        dummy_market_features,
        dummy_industry_ids,
        dummy_stock_fund_features,
        dummy_stock_intraday_ts_features,
    )
    logger.info(output)
