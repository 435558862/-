"""Crash-safe local diagnostics for customer installations."""

import faulthandler
import logging
import platform
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.version import PRODUCT_TITLE


def setup_runtime_diagnostics(root: Path) -> Path:
    log_dir = root / 'storage' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / 'app.log'
    handler = RotatingFileHandler(
        log_path, maxBytes=2 * 1024 * 1024, backupCount=4, encoding='utf-8',
    )
    handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s %(name)s %(message)s',
    ))
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if not any(
        isinstance(item, RotatingFileHandler)
        and Path(item.baseFilename) == log_path.resolve()
        for item in root_logger.handlers
    ):
        root_logger.addHandler(handler)
    crash_stream = (log_dir / 'fatal.log').open('a', encoding='utf-8')
    faulthandler.enable(file=crash_stream, all_threads=True)

    previous_hook = sys.excepthook

    def report_exception(exc_type, exc_value, traceback):
        logging.getLogger('crash').critical(
            '未捕获异常', exc_info=(exc_type, exc_value, traceback),
        )
        previous_hook(exc_type, exc_value, traceback)

    sys.excepthook = report_exception
    logging.getLogger(__name__).info(
        '启动 %s | Python %s | %s %s', PRODUCT_TITLE,
        platform.python_version(), platform.system(), platform.release(),
    )
    return log_path

