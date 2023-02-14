import logging


def get_logger(name, log_level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger
