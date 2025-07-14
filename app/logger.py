import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

def setup_logger(name: str = "model_manager", log_dir: str = "logs") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger 

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    stream_handler = logging.StreamHandler()
    stream_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s - %(message)s")
    stream_handler.setFormatter(stream_formatter)

    file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
    file_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger
