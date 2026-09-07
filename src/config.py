import logging
from pathlib import Path


def setup_logger(
    name: str = "YOLOTrainer", level: int = logging.INFO, log_file: str | None = None
) -> logging.Logger:
    """Create and configure a logger instance."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    if not any(type(handler) is logging.StreamHandler for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file and not any(
        isinstance(handler, logging.FileHandler)
        and handler.baseFilename == str(Path(log_file).resolve())
        for handler in logger.handlers
    ):
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
