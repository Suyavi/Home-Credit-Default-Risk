"""Load and validate pipeline YAML configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    seed: int


@dataclass(frozen=True)
class LoggingConfig:
    level: str


@dataclass(frozen=True)
class DataConfig:
    raw_dir: str
    processed_dir: str


@dataclass(frozen=True)
class TrainingConfig:
    n_folds: int
    num_boost_round: int
    early_stopping_rounds: int
    log_evaluation_period: int


@dataclass(frozen=True)
class ArtifactsConfig:
    runs_dir: str


@dataclass(frozen=True)
class OutputConfig:
    models_dir: str
    reports_dir: str
    model_filename: str
    submission_filename: str
    cv_report_filename: str
    feature_importance_filename: str


@dataclass(frozen=True)
class PipelineBehaviorConfig:
    export_features: bool
    strict_validation: bool


@dataclass(frozen=True)
class PipelineConfig:
    project: ProjectConfig
    logging: LoggingConfig
    data: DataConfig
    training: TrainingConfig
    model_params: dict[str, Any]  # LightGBM overrides (see default.yaml `model`)
    artifacts: ArtifactsConfig
    output: OutputConfig
    pipeline: PipelineBehaviorConfig


def _get(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_pipeline_config(path: Path) -> PipelineConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    project = ProjectConfig(seed=int(_get(raw, "project", "seed", default=42)))
    logging_cfg = LoggingConfig(level=str(_get(raw, "logging", "level", default="INFO")).upper())
    data = DataConfig(
        raw_dir=str(_get(raw, "data", "raw_dir", default="data/raw")),
        processed_dir=str(_get(raw, "data", "processed_dir", default="data/processed")),
    )
    training = TrainingConfig(
        n_folds=int(_get(raw, "training", "n_folds", default=5)),
        num_boost_round=int(_get(raw, "training", "num_boost_round", default=15_000)),
        early_stopping_rounds=int(_get(raw, "training", "early_stopping_rounds", default=200)),
        log_evaluation_period=int(_get(raw, "training", "log_evaluation_period", default=200)),
    )
    artifacts = ArtifactsConfig(runs_dir=str(_get(raw, "artifacts", "runs_dir", default="artifacts/runs")))
    output = OutputConfig(
        models_dir=str(_get(raw, "output", "models_dir", default="models")),
        reports_dir=str(_get(raw, "output", "reports_dir", default="reports")),
        model_filename=str(_get(raw, "output", "model_filename", default="trained_model.pkl")),
        submission_filename=str(_get(raw, "output", "submission_filename", default="submission.csv")),
        cv_report_filename=str(_get(raw, "output", "cv_report_filename", default="cv_summary.md")),
        feature_importance_filename=str(
            _get(raw, "output", "feature_importance_filename", default="feature_importance_top.png")
        ),
    )
    pipeline = PipelineBehaviorConfig(
        export_features=bool(_get(raw, "pipeline", "export_features", default=False)),
        strict_validation=bool(_get(raw, "pipeline", "strict_validation", default=True)),
    )

    raw_model = raw.get("model")
    model_params: dict[str, Any] = {}
    if isinstance(raw_model, dict):
        model_params = {str(k): v for k, v in raw_model.items() if v is not None}

    return PipelineConfig(
        project=project,
        logging=logging_cfg,
        data=data,
        training=training,
        model_params=model_params,
        artifacts=artifacts,
        output=output,
        pipeline=pipeline,
    )
