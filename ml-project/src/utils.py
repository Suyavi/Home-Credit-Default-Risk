"""Shared paths, constants, and helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Project root = parent of `src/`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

SEED = 42
N_FOLDS = 5


def reduce_memory(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            c_min, c_max = df[col].min(), df[col].max()
            if pd.isnull(c_min) or pd.isnull(c_max):
                continue
            if pd.api.types.is_integer_dtype(df[col]):
                for dtype in [np.int8, np.int16, np.int32, np.int64]:
                    if np.iinfo(dtype).min <= c_min <= c_max <= np.iinfo(dtype).max:
                        df[col] = df[col].astype(dtype)
                        break
            else:
                for dtype in [np.float32, np.float64]:
                    if np.finfo(dtype).min <= c_min <= c_max <= np.finfo(dtype).max:
                        df[col] = df[col].astype(dtype)
                        break
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        pct = 100 * (start_mem - end_mem) / start_mem if start_mem else 0
        print(f"Memory: {start_mem:.1f}MB → {end_mem:.1f}MB ({pct:.1f}% saved)")
    return df


def ensure_dirs() -> None:
    for p in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR):
        p.mkdir(parents=True, exist_ok=True)
