import typing

if typing.TYPE_CHECKING:
    from loguru import Logger
else:
    Logger = None

__all__ = ["logger"]


def __get_logger() -> Logger:
    from loguru import logger

    return logger


logger = __get_logger()


