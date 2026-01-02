import logging
import os
from datetime import datetime

_LOGGER = None


def get_logger():
    global _LOGGER
    if _LOGGER:
        return _LOGGER

    logger = logging.getLogger("FacialRecognitionEngine")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        os.makedirs("logs", exist_ok=True)

        today = datetime.now().strftime("%Y-%m-%d")
        log_file = f"logs/facial_recognition_{today}.log"

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)

        # File handler (date in filename)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        logger.addHandler(ch)
        logger.addHandler(fh)

    _LOGGER = logger
    return logger
