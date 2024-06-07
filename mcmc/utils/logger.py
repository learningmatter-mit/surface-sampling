import logging


def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter("%(asctime)s - %(name)s | %(message)s", "%H:%M:%S")
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    return logger
