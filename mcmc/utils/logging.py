"""Logging utilities."""

import logging


def setup_logger(name, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """Set up a logger.

    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file.
        level (int, optional): Logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: Logger object.
    """
    formatter = logging.Formatter("%(asctime)s - %(name)s | %(levelname)s: %(message)s", "%H:%M:%S")

    # log to file
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(formatter)

    # log to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


class SilenceLogger:
    """Context manager for silencing logger output."""

    def __enter__(self):
        """Silence logger output."""
        logging.disable(logging.CRITICAL)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Enable logger output."""
        logging.disable(logging.NOTSET)
        return False
