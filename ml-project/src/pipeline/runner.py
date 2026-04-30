"""Orchestrates validation → features → training → run manifest."""

from __future__ import annotations

import gc
import json
import random
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_preprocessing import drop_and_encode_for_lgb, load_raw_tables, save_processed
from evaluation import oof_roc_auc, plot_top_feature_importance, summarize_feature_importance, write_cv_report
from feature_engineering import build_enriched_train_test
from model_training import save_model, save_submission, train_cv_lightgbm
from pipeline.config import load_pipeline_config
from pipeline.context import RunContext, build_context
from pipeline.logging_utils import setup_logging
from pipeline.steps import step
from pipeline.validation import validate_raw_directory


@dataclass(frozen=True)
class PipelineResult:
    run_id: str
    run_dir: Path
    manifest_path: Path
    cv_auc: float
    fold_scores: list[float]
    n_features: int
    model_path: Path
    submission_path: Path


def _try_git_sha(root: Path) -> tuple[str | None, bool | None]:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if r.returncode != 0:
            return None, None
        sha = r.stdout.strip() or None
        r2 = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        dirty: bool | None
        if r2.returncode == 0:
            dirty = bool(r2.stdout.strip())
        else:
            dirty = None
        return sha, dirty
    except (OSError, subprocess.TimeoutExpired):
        return None, None


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _ensure_output_dirs(ctx: RunContext) -> None:
    for p in (ctx.models_dir(), ctx.reports_dir(), ctx.figures_dir(), ctx.processed_dir(), ctx.run_dir):
        p.mkdir(parents=True, exist_ok=True)


def _write_manifest(
    ctx: RunContext,
    *,
    cv_auc: float,
    fold_scores: list[float],
    n_features: int,
    model_path: Path,
    submission_path: Path,
    python_executable: str,
    mean_train_val_gap: float | None = None,
    fold_val_std: float | None = None,
    mean_best_iteration: float | None = None,
) -> Path:
    finished_at = datetime.now(timezone.utc)
    git_sha, git_dirty = _try_git_sha(ctx.root)
    body: dict[str, Any] = {
        "run_id": ctx.run_id,
        "started_at": ctx.started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "config_path": str(ctx.config_path.as_posix()),
        "config": asdict(ctx.config),
        "metrics": {
            "oof_roc_auc": cv_auc,
            "fold_val_roc_aucs": fold_scores,
            "n_features": n_features,
            "mean_train_val_auc_gap": mean_train_val_gap,
            "fold_val_auc_std": fold_val_std,
            "mean_best_boosting_iteration": mean_best_iteration,
        },
        "artifacts": {
            "model_path": str(model_path.relative_to(ctx.root).as_posix()),
            "submission_path": str(submission_path.relative_to(ctx.root).as_posix()),
        },
        "environment": {
            "python": python_executable,
            "python_version": sys.version.split()[0],
        },
        "git": {"commit": git_sha, "dirty": git_dirty},
    }
    out = ctx.run_dir / "manifest.json"
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return out


def validate_only(*, project_root: Path, config_path: Path) -> None:
    root = project_root.resolve()
    cfg = load_pipeline_config(config_path.resolve())
    logger = setup_logging(cfg.logging.level)
    validate_raw_directory(
        root / cfg.data.raw_dir,
        strict=cfg.pipeline.strict_validation,
        logger=logger,
    )
    logger.info("Validation-only run completed successfully.")


def run_pipeline(
    *,
    project_root: Path,
    config_path: Path,
    run_id: str | None = None,
) -> PipelineResult:
    project_root = project_root.resolve()
    cfg = load_pipeline_config(config_path.resolve())
    logger = setup_logging(cfg.logging.level)
    ctx = build_context(project_root, cfg, config_path.resolve(), logger, run_id=run_id)

    log = ctx.logger
    log.info("Starting pipeline | run_id=%s", ctx.run_id)
    log.info("Using config file: %s", ctx.config_path)

    _set_global_seed(cfg.project.seed)
    _ensure_output_dirs(ctx)
    shutil.copy2(ctx.config_path, ctx.run_dir / "config_resolved.yaml")

    with step(log, "validate_raw"):
        validate_raw_directory(
            ctx.raw_dir(),
            strict=cfg.pipeline.strict_validation,
            logger=log,
        )

    with step(log, "load_raw"):
        raw = load_raw_tables(ctx.raw_dir(), verbose=False)

    with step(log, "feature_engineering"):
        app_train, app_test = build_enriched_train_test(raw)
        log.info("Feature engineering output shapes: train=%s test=%s", app_train.shape, app_test.shape)

    del raw
    gc.collect()

    if cfg.pipeline.export_features:
        with step(log, "export_features"):
            save_processed(app_train, app_test, ctx.processed_dir())

    with step(log, "matrix_prep"):
        X_train, X_test, y, _train_ids, test_ids = drop_and_encode_for_lgb(
            app_train,
            app_test,
            enable_target_encoding=True,
            te_max_cols=8,
            te_smoothing=80.0,
            te_n_splits=cfg.training.n_folds,
            # Pair target encoding can overfit on Home Credit with wide feature space.
            enable_pair_target_encoding=False,
            te_pair_pool_cols=6,
            te_pair_max_pairs=4,
            te_pair_max_unique=120,
            te_pair_smoothing=80.0,
            random_state=cfg.project.seed,
        )

    del app_train, app_test
    gc.collect()

    with step(log, "train_cv"):
        seed_list = [cfg.project.seed, cfg.project.seed + 77, cfg.project.seed + 2027]
        seed_list_str = ",".join(str(s) for s in seed_list)
        outs = []
        for s in seed_list:
            log.info("Training seed=%s", s)
            out = train_cv_lightgbm(
                X_train,
                y,
                X_test,
                lgb_param_overrides=cfg.model_params,
                n_folds=cfg.training.n_folds,
                random_state=s,
                num_boost_round=cfg.training.num_boost_round,
                early_stopping_rounds=cfg.training.early_stopping_rounds,
                log_evaluation_period=cfg.training.log_evaluation_period,
            )
            outs.append(out)
            log.info(
                "Seed %s diagnostics: oof_auc=%.5f gap=%.5f std=%.5f best_iter=%.1f",
                s,
                out.oof_auc,
                out.mean_train_val_gap,
                out.fold_val_auc_std,
                out.mean_best_iteration,
            )

        test_pred = np.mean(np.vstack([o.test_preds for o in outs]), axis=0)
        oof_ens = np.mean(np.vstack([o.oof_preds for o in outs]), axis=0)
        cv_auc = float(oof_roc_auc(y, oof_ens))
        fold_scores = np.mean(np.vstack([o.fold_val_aucs for o in outs]), axis=0).tolist()
        fold_train_aucs = np.mean(np.vstack([o.fold_train_aucs for o in outs]), axis=0).tolist()
        mean_train_val_gap = float(np.mean(np.array(fold_train_aucs) - np.array(fold_scores)))
        fold_val_std = float(np.std(fold_scores))
        mean_best_iteration = float(np.mean([o.mean_best_iteration for o in outs]))
        feat_imp = pd.concat([o.feature_importance for o in outs], axis=0, ignore_index=True)
        tr_out = max(outs, key=lambda o: o.oof_auc)
        model = tr_out.model
        log.info(
            "Ensemble diagnostics: oof_auc=%.5f mean_gap=%.5f fold_val_std=%.5f mean_best_iter=%.1f",
            cv_auc,
            mean_train_val_gap,
            fold_val_std,
            mean_best_iteration,
        )

    model_path = ctx.models_dir() / cfg.output.model_filename
    submission_path = ctx.root / cfg.output.submission_filename

    with step(log, "export_artifacts"):
        save_model(model, model_path)
        save_submission(test_ids, test_pred, submission_path)
        shutil.copy2(submission_path, ctx.run_dir / cfg.output.submission_filename)

        summary = summarize_feature_importance(feat_imp, top_n=30)
        log.info("Top features (head):\n%s", summary.head(10).to_string())
        fi_path = ctx.figures_dir() / cfg.output.feature_importance_filename
        plot_top_feature_importance(summary, save_path=fi_path)

        git_sha, _ = _try_git_sha(ctx.root)
        extra_meta = {
            "run_id": ctx.run_id,
            "config": ctx.config_path.name,
            "git_commit": git_sha or "unknown",
            "ensemble_seeds": seed_list_str,
        }
        cv_report_path = ctx.reports_dir() / cfg.output.cv_report_filename
        write_cv_report(
            cv_report_path,
            cv_auc,
            fold_scores,
            X_train.shape[1],
            random_seed=cfg.project.seed,
            fold_train_aucs=fold_train_aucs,
            mean_train_val_gap=mean_train_val_gap,
            fold_val_std=fold_val_std,
            mean_best_iteration=mean_best_iteration,
            extra_metadata=extra_meta,
        )
        shutil.copy2(cv_report_path, ctx.run_dir / cfg.output.cv_report_filename)

    manifest_path = _write_manifest(
        ctx,
        cv_auc=cv_auc,
        fold_scores=fold_scores,
        n_features=int(X_train.shape[1]),
        model_path=model_path,
        submission_path=submission_path,
        python_executable=sys.executable,
        mean_train_val_gap=mean_train_val_gap,
        fold_val_std=fold_val_std,
        mean_best_iteration=mean_best_iteration,
    )
    log.info("Wrote run manifest: %s", manifest_path)
    log.info("Pipeline finished successfully.")

    return PipelineResult(
        run_id=ctx.run_id,
        run_dir=ctx.run_dir,
        manifest_path=manifest_path,
        cv_auc=cv_auc,
        fold_scores=fold_scores,
        n_features=int(X_train.shape[1]),
        model_path=model_path,
        submission_path=submission_path,
    )
