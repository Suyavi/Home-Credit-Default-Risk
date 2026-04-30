"""Timed pipeline steps (context managers)."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def step(logger: logging.Logger, name: str) -> Iterator[None]:
    t0 = time.perf_counter()
    logger.info("BEGIN %s", name)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        logger.info("END %s (%.2fs)", name, elapsed)
