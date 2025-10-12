import logging
from typing import Optional

def setup_logger(
    name: str = "YOLOTrainer",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """Create and configure a logger instance."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
