import sys
from pathlib import Path

from loguru import logger

_stdout_configured = False


def configure_stdout_logging() -> None:
    """Replace loguru's default stderr sink with stdout. Idempotent."""
    global _stdout_configured
    if _stdout_configured:
        return
    logger.remove()
    logger.add(sys.stdout, colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
    _stdout_configured = True


def add_file_sink(log_file: Path) -> int:
    """Add an append-only file sink. Returns the sink id for later removal."""
    return logger.add(
        str(log_file),
        mode="a",
        colorize=False,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    )
