"""Structured console logging for pipeline runs."""

from __future__ import annotations

import logging
import sys
from typing import TextIO


def setup_logging(level: str = "INFO", stream: TextIO | None = None) -> logging.Logger:
    log = logging.getLogger("ml_project")
    log.handlers.clear()
    log.setLevel(getattr(logging, level.upper(), logging.INFO))
    log.propagate = False

    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setLevel(log.level)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    log.addHandler(handler)
    return log
