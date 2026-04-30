"""LightGBM stratified CV training and artifact export."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from evaluation import oof_roc_auc
from utils import MODELS_DIR, N_FOLDS, SEED


@dataclass(frozen=True)
class CVTrainResult:
    """Training outputs plus diagnostics for bias–variance checks."""

    oof_preds: np.ndarray
    test_preds: np.ndarray
    feature_importance: pd.DataFrame
    oof_auc: float
    fold_val_aucs: list[float]
    fold_train_aucs: list[float]
    mean_train_val_gap: float
    fold_val_auc_std: float
    mean_best_iteration: float
    model: lgb.Booster


def default_lgb_params(random_state: int | None = None) -> dict:
    """
    Conservative tree defaults: subsampling + L2 + min_leaf mass reduce overfitting;
    early stopping + many rounds address underfitting if signal exists.
    """
    rs = SEED if random_state is None else random_state
    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.02,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.70,
        "bagging_freq": 5,
        "min_child_samples": 70,
        "min_child_weight": 0.001,
        "reg_alpha": 0.50,
        "reg_lambda": 0.52,
        "max_depth": 7,
        "min_gain_to_split": 0.0,
        "random_state": rs,
        "n_jobs": -1,
        "verbose": -1,
    }


def build_lgb_params(random_state: int | None, overrides: dict[str, object] | None) -> dict:
    """Merge YAML `model` dict onto defaults (YAML wins)."""
    base = default_lgb_params(random_state).copy()
    if not overrides:
        return base
    for k, v in overrides.items():
        if v is None or k in ("objective", "metric", "boosting_type", "device_type"):
            continue
        base[k] = v
    return base


def train_cv_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    lgb_param_overrides: dict[str, object] | None = None,
    n_folds: int = N_FOLDS,
    random_state: int | None = None,
    num_boost_round: int = 15000,
    early_stopping_rounds: int = 200,
    log_evaluation_period: int = 200,
) -> CVTrainResult:
    """
    Stratified K-fold with early stopping on a validation fold.

    Each fold logs train vs validation ROC-AUC at ``best_iteration`` so you can
    spot overfitting (large train–val gap) or instability (high std across folds).
    """
    rs = SEED if random_state is None else random_state
    params = build_lgb_params(rs, lgb_param_overrides)
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rs)
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    feature_importance = pd.DataFrame()
    last_model: lgb.Booster | None = None
    fold_val_aucs: list[float] = []
    fold_train_aucs: list[float] = []
    best_iterations: list[int] = []

    for fold, (train_idx, val_idx) in enumerate(folds.split(X_train, y_train), 1):
        print(f"\n--- Fold {fold}/{n_folds} ---")
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(log_evaluation_period),
            ],
        )

        bi = int(model.best_iteration) if model.best_iteration is not None else num_boost_round
        best_iterations.append(bi)

        tr_pred = model.predict(X_tr, num_iteration=bi)
        val_pred = model.predict(X_val, num_iteration=bi)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(X_test, num_iteration=bi) / n_folds

        train_auc = float(roc_auc_score(y_tr, tr_pred))
        val_auc = float(roc_auc_score(y_val, val_pred))
        fold_train_aucs.append(train_auc)
        fold_val_aucs.append(val_auc)
        gap = train_auc - val_auc
        print(f"Fold {fold} train AUC {train_auc:.5f} | val AUC {val_auc:.5f} | gap {gap:+.5f} | best_iter {bi}")

        fold_imp = pd.DataFrame(
            {
                "feature": X_train.columns,
                "importance": model.feature_importance(importance_type="gain"),
                "fold": fold,
            }
        )
        feature_importance = pd.concat([feature_importance, fold_imp], axis=0)
        last_model = model

    oof_auc = oof_roc_auc(y_train, oof_preds)
    mean_gap = float(np.mean(np.array(fold_train_aucs) - np.array(fold_val_aucs)))
    fold_std = float(np.std(fold_val_aucs))
    mean_best = float(np.mean(best_iterations))
    print(f"\nCV ROC-AUC (OOF): {oof_auc:.5f}")
    print(f"Mean train–val gap: {mean_gap:+.5f} (positive ⇒ train higher; watch large gaps for overfitting)")
    print(f"Std dev of fold val AUCs: {fold_std:.5f} (high ⇒ unstable across splits)")
    print(f"Mean best iteration: {mean_best:.1f}")
    assert last_model is not None
    return CVTrainResult(
        oof_preds=oof_preds,
        test_preds=test_preds,
        feature_importance=feature_importance,
        oof_auc=oof_auc,
        fold_val_aucs=fold_val_aucs,
        fold_train_aucs=fold_train_aucs,
        mean_train_val_gap=mean_gap,
        fold_val_auc_std=fold_std,
        mean_best_iteration=mean_best,
        model=last_model,
    )


def save_model(model: lgb.Booster, path: Path | None = None) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = path or (MODELS_DIR / "trained_model.pkl")
    with open(out, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model: {out}")
    return out


def save_submission(test_ids: pd.Series, test_preds: np.ndarray, path: Path) -> Path:
    sub = pd.DataFrame({"SK_ID_CURR": test_ids, "TARGET": test_preds})
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    print(f"Saved submission: {path}")
    return path
