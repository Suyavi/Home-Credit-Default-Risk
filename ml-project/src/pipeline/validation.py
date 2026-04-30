"""Raw data presence and minimal schema checks."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

REQUIRED_RAW_FILES = (
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "POS_CASH_balance.csv",
    "credit_card_balance.csv",
    "installments_payments.csv",
)

MIN_APPLICATION_TRAIN_COLUMNS = frozenset({"SK_ID_CURR", "TARGET"})


def validate_raw_directory(raw_dir: Path, *, strict: bool, logger: logging.Logger) -> None:
    raw_dir = raw_dir.resolve()
    missing = [name for name in REQUIRED_RAW_FILES if not (raw_dir / name).exists()]
    if missing:
        msg = f"Missing raw files under {raw_dir}: {missing}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    if strict:
        train_path = raw_dir / "application_train.csv"
        head = pd.read_csv(train_path, nrows=1)
        cols = set(head.columns)
        if not MIN_APPLICATION_TRAIN_COLUMNS.issubset(cols):
            msg = (
                f"{train_path} must contain columns {MIN_APPLICATION_TRAIN_COLUMNS}; "
                f"found sample columns {sorted(cols)[:20]}..."
            )
            logger.error(msg)
            raise ValueError(msg)

    logger.info("Raw data validation passed for %s", raw_dir)
