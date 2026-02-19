import logging 
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOGGER = logging.getLogger("polnet")

def setup_logger(log_folder: Path, log_level: int = logging.DEBUG):
    """Set up the logger with console and file handlers.

    Returns:
        None
    """
    if _LOGGER.hasHandlers():
        _LOGGER.warning("Logger already set up")
        return 
    log_folder.mkdir(parents=True, exist_ok=True)
    log_file = log_folder / "polnet.log"

    # Create root logger
    _LOGGER.setLevel(log_level)

    # --- Console Handler ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        '[%(asctime)s] %(message)s',
        datefmt='%H:%M'
    )
    console_handler.setFormatter(console_format)

    # --- File Handler (rotating) ---
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5_000_000, backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter(
        '%(asctime)s,%(msecs)03d | %(levelname)s | %(name)s | %(process)d:%(threadName)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)

    # Add handlers
    _LOGGER.addHandler(console_handler)
    _LOGGER.addHandler(file_handler)