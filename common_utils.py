import argparse
import logging
import os
from prepare_math_reasoning_data import MATH_REASONING_DATA_NAME

formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
BASE_LOG_PATH = "/tmp/lmsmall/logs"

def setup_logger(class_name:str, model_name: str, level=logging.INFO):
    """To setup as many loggers as you want"""
    os.makedirs(BASE_LOG_PATH, exist_ok=True)
    log_file_path = os.path.join(BASE_LOG_PATH, f"{model_name}_log.txt")
    handler = logging.FileHandler(log_file_path)
    handler.setFormatter(formatter)
    logger = logging.getLogger(class_name)
    logger.setLevel(level)
    logger.addHandler(handler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    return (log_file_path, logger)


def setup_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-log",
        "--loglevel",
        default="INFO",
    )
    parser.add_argument(
        "--data_name",
        default=MATH_REASONING_DATA_NAME,
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=False,
    )
    return parser
