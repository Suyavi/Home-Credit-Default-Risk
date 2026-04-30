"""Load raw CSVs from ``data/raw`` and prepare matrices for modeling."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from utils import DATA_RAW, reduce_memory


@dataclass
class RawTables:
    application_train: pd.DataFrame
    application_test: pd.DataFrame
    bureau: pd.DataFrame
    bureau_balance: pd.DataFrame
    previous_application: pd.DataFrame
    pos_cash_balance: pd.DataFrame
    credit_card_balance: pd.DataFrame
    installments_payments: pd.DataFrame


def load_raw_tables(raw_dir: Path | None = None, verbose: bool = True) -> RawTables:
    """Load Home Credit CSVs from ``data/raw`` (Kaggle-style filenames)."""
    d = Path(raw_dir) if raw_dir is not None else DATA_RAW
    if verbose:
        print(f"Loading from {d} ...")
    return RawTables(
        application_train=reduce_memory(pd.read_csv(d / "application_train.csv"), verbose),
        application_test=reduce_memory(pd.read_csv(d / "application_test.csv"), verbose),
        bureau=reduce_memory(pd.read_csv(d / "bureau.csv"), verbose),
        bureau_balance=reduce_memory(pd.read_csv(d / "bureau_balance.csv"), verbose),
        previous_application=reduce_memory(pd.read_csv(d / "previous_application.csv"), verbose),
        pos_cash_balance=reduce_memory(pd.read_csv(d / "POS_CASH_balance.csv"), verbose),
        credit_card_balance=reduce_memory(pd.read_csv(d / "credit_card_balance.csv"), verbose),
        installments_payments=reduce_memory(pd.read_csv(d / "installments_payments.csv"), verbose),
    )


def drop_and_encode_for_lgb(
    app_train: pd.DataFrame,
    app_test: pd.DataFrame,
    *,
    enable_target_encoding: bool = True,
    te_max_cols: int = 15,
    te_min_unique: int = 3,
    te_max_unique: int = 200,
    te_smoothing: float = 30.0,
    te_n_splits: int = 5,
    enable_pair_target_encoding: bool = True,
    te_pair_pool_cols: int = 6,
    te_pair_max_pairs: int = 4,
    te_pair_max_unique: int = 120,
    te_pair_smoothing: float = 60.0,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Strip IDs/target, label-encode object columns consistently on train+test.

    Returns
    -------
    X_train, X_test, y, train_ids, test_ids
    """
    train_labels = app_train["TARGET"]
    train_ids = app_train["SK_ID_CURR"]
    test_ids = app_test["SK_ID_CURR"]

    X_train = app_train.drop(columns=["TARGET", "SK_ID_CURR"])
    X_test = app_test.drop(columns=["SK_ID_CURR"])
    cat_cols: list[str] = []

    for col in X_train.columns:
        if X_train[col].dtype == "object" or X_train[col].dtype.name == "category":
            cat_cols.append(col)
            le = LabelEncoder()
            le.fit(pd.concat([X_train[col], X_test[col]]).astype(str))
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))

    if enable_target_encoding and cat_cols:
        _add_oof_target_encoding_features(
            X_train,
            X_test,
            train_labels,
            cat_cols=cat_cols,
            max_cols=te_max_cols,
            min_unique=te_min_unique,
            max_unique=te_max_unique,
            smoothing=te_smoothing,
            n_splits=te_n_splits,
            random_state=random_state,
        )
        if enable_pair_target_encoding:
            _add_oof_pair_target_encoding_features(
                X_train,
                X_test,
                train_labels,
                cat_cols=cat_cols,
                pair_pool_cols=te_pair_pool_cols,
                max_pairs=te_pair_max_pairs,
                max_unique=te_pair_max_unique,
                smoothing=te_pair_smoothing,
                n_splits=te_n_splits,
                random_state=random_state,
            )

    return X_train, X_test, train_labels, train_ids, test_ids


def _add_oof_target_encoding_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y: pd.Series,
    *,
    cat_cols: list[str],
    max_cols: int,
    min_unique: int,
    max_unique: int,
    smoothing: float,
    n_splits: int,
    random_state: int,
) -> None:
    """Leakage-safe OOF target encoding for selected categorical columns."""
    if n_splits < 2:
        return
    y = y.reset_index(drop=True)
    global_mean = float(y.mean())

    # Prioritize richer categoricals while avoiding extremely high-cardinality noise.
    candidates: list[tuple[str, int]] = []
    for c in cat_cols:
        nun = int(X_train[c].nunique(dropna=False))
        if min_unique <= nun <= max_unique:
            candidates.append((c, nun))
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected_cols = [c for c, _ in candidates[:max_cols]]
    if not selected_cols:
        return

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_arr = y.to_numpy()

    for col in selected_cols:
        tr_col = X_train[col]
        te_train = np.full(len(X_train), global_mean, dtype=np.float64)

        for fit_idx, val_idx in skf.split(X_train, y_arr):
            fit_series = tr_col.iloc[fit_idx]
            fit_y = y.iloc[fit_idx]
            stats = fit_y.groupby(fit_series).agg(["mean", "count"])
            smooth = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
            te_train[val_idx] = tr_col.iloc[val_idx].map(smooth).fillna(global_mean).to_numpy(dtype=np.float64)

        full_stats = y.groupby(tr_col).agg(["mean", "count"])
        full_smooth = (
            full_stats["mean"] * full_stats["count"] + global_mean * smoothing
        ) / (full_stats["count"] + smoothing)
        te_test = X_test[col].map(full_smooth).fillna(global_mean).to_numpy(dtype=np.float64)

        X_train[f"{col}_TE"] = te_train
        X_test[f"{col}_TE"] = te_test


def _add_oof_pair_target_encoding_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y: pd.Series,
    *,
    cat_cols: list[str],
    pair_pool_cols: int,
    max_pairs: int,
    max_unique: int,
    smoothing: float,
    n_splits: int,
    random_state: int,
) -> None:
    """Leakage-safe OOF target encoding on selected categorical pairs."""
    if n_splits < 2 or max_pairs <= 0 or pair_pool_cols < 2:
        return
    y = y.reset_index(drop=True)
    global_mean = float(y.mean())
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_arr = y.to_numpy()

    ranked = sorted(cat_cols, key=lambda c: int(X_train[c].nunique(dropna=False)), reverse=True)
    pool = ranked[:pair_pool_cols]
    if len(pool) < 2:
        return

    selected_pairs: list[tuple[str, str, int]] = []
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            c1, c2 = pool[i], pool[j]
            pair_key = X_train[c1].astype(str) + "|" + X_train[c2].astype(str)
            nun = int(pair_key.nunique(dropna=False))
            if 5 <= nun <= max_unique:
                selected_pairs.append((c1, c2, nun))
    if not selected_pairs:
        return
    selected_pairs.sort(key=lambda x: x[2], reverse=True)

    for c1, c2, _ in selected_pairs[:max_pairs]:
        tr_key = X_train[c1].astype(str) + "|" + X_train[c2].astype(str)
        te_train = np.full(len(X_train), global_mean, dtype=np.float64)

        for fit_idx, val_idx in skf.split(X_train, y_arr):
            fit_key = tr_key.iloc[fit_idx]
            fit_y = y.iloc[fit_idx]
            stats = fit_y.groupby(fit_key).agg(["mean", "count"])
            smooth = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
            te_train[val_idx] = tr_key.iloc[val_idx].map(smooth).fillna(global_mean).to_numpy(dtype=np.float64)

        full_stats = y.groupby(tr_key).agg(["mean", "count"])
        full_smooth = (
            full_stats["mean"] * full_stats["count"] + global_mean * smoothing
        ) / (full_stats["count"] + smoothing)
        te_test_key = X_test[c1].astype(str) + "|" + X_test[c2].astype(str)
        te_test = te_test_key.map(full_smooth).fillna(global_mean).to_numpy(dtype=np.float64)

        out_col = f"{c1}_{c2}_TE2"
        X_train[out_col] = te_train
        X_test[out_col] = te_test


def save_processed(train: pd.DataFrame, test: pd.DataFrame, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    train.to_csv(directory / "train_features.csv", index=False)
    test.to_csv(directory / "test_features.csv", index=False)
    print(f"Saved processed features to {directory}")


def free_raw_tables(t: RawTables) -> None:
    del t
    gc.collect()
