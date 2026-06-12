import sys
from loguru import logger
from shared.config import settings


def setup_logging(service_name: str = "mogao"):
    logger.remove()

    fmt = (
        f"<green>{{time:YYYY-MM-DD HH:mm:ss}}</green> "
        f"| <magenta>{service_name:<14}</magenta> "
        f"| <level>{{level:<8}}</level> "
        f"| <cyan>{{name}}</cyan>:<cyan>{{function}}</cyan>:<cyan>{{line}}</cyan> "
        f"| <level>{{message}}</level>"
    )

    logger.add(
        sys.stdout,
        format=fmt,
        level="DEBUG" if settings.DEBUG else "INFO",
        colorize=True,
        enqueue=True,
        backtrace=True,
        diagnose=settings.DEBUG,
    )

    logger.add(
        f"logs/{service_name}_{{time:YYYY-MM-DD}}.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | {message}",
        level="INFO",
        rotation="100 MB",
        retention="30 days",
        compression="gz",
        enqueue=True,
        encoding="utf-8",
    )

    logger.add(
        f"logs/{service_name}_error_{{time:YYYY-MM-DD}}.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | {message}",
        level="ERROR",
        rotation="50 MB",
        retention="60 days",
        compression="gz",
        enqueue=True,
        encoding="utf-8",
    )

    return logger
