"""Metrics, CV summaries, and feature-importance plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from utils import FIGURES_DIR, SEED


def oof_roc_auc(y_true: pd.Series | np.ndarray, oof_preds: np.ndarray) -> float:
    return float(roc_auc_score(y_true, oof_preds))


def fold_roc_auc(y_true: pd.Series, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score))


def summarize_feature_importance(feature_importance: pd.DataFrame, top_n: int = 30) -> pd.Series:
    """Mean gain importance across folds."""
    return feature_importance.groupby("feature")["importance"].mean().sort_values(ascending=False).head(top_n)


def plot_top_feature_importance(
    importance_summary: pd.Series,
    title: str = "Top feature importance (mean gain)",
    save_path: Path | None = None,
    top_n: int = 20,
) -> None:
    top = importance_summary.head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.25)))
    top.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    out = save_path or (FIGURES_DIR / "feature_importance_top.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out}")


def write_cv_report(
    path: Path,
    cv_auc: float,
    fold_scores: list[float],
    n_features: int,
    *,
    random_seed: int | None = None,
    fold_train_aucs: list[float] | None = None,
    mean_train_val_gap: float | None = None,
    fold_val_std: float | None = None,
    mean_best_iteration: float | None = None,
    extra_metadata: dict[str, str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seed_line = random_seed if random_seed is not None else SEED
    lines = [
        "# Modeling summary",
        "",
        f"- **Random seed**: {seed_line}",
        f"- **OOF ROC-AUC**: {cv_auc:.5f}",
        f"- **Fold validation ROC-AUCs**: {fold_scores}",
        f"- **Feature count**: {n_features}",
        "",
    ]
    if fold_train_aucs is not None and mean_train_val_gap is not None and fold_val_std is not None:
        lines.extend(
            [
                "## Generalization (per-fold, at early-stopping iteration)",
                "",
                f"- **Fold train ROC-AUCs**: {fold_train_aucs}",
                f"- **Mean (train − val) AUC gap**: {mean_train_val_gap:+.5f}",
                "  - Small gap: train and validation align (healthy).",
                "  - Large positive gap: model fits train much better than val (risk of overfitting).",
                "  - Large negative gap: unusual; check data or splits.",
                f"- **Std dev of fold val AUCs**: {fold_val_std:.5f}",
                "  - High std: unstable across folds (variance); consider more regularization or data checks.",
                "",
            ]
        )
        if mean_best_iteration is not None:
            lines.append(f"- **Mean best boosting iteration**: {mean_best_iteration:.1f}")
            lines.append("")
    if extra_metadata:
        lines.append("## Run metadata")
        lines.append("")
        for k, v in sorted(extra_metadata.items()):
            lines.append(f"- **{k}**: {v}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report snippet: {path}")
