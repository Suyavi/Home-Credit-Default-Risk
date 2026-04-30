"""Per-run paths and resolved configuration."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import PipelineConfig


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


@dataclass
class RunContext:
    """Single training run: isolated artifact directory + logger."""

    root: Path
    run_id: str
    started_at: datetime
    config: PipelineConfig
    logger: logging.Logger
    config_path: Path

    @property
    def run_dir(self) -> Path:
        return self.root / self.config.artifacts.runs_dir / self.run_id

    def raw_dir(self) -> Path:
        return self.root / self.config.data.raw_dir

    def processed_dir(self) -> Path:
        return self.root / self.config.data.processed_dir

    def models_dir(self) -> Path:
        return self.root / self.config.output.models_dir

    def reports_dir(self) -> Path:
        return self.root / self.config.output.reports_dir

    def figures_dir(self) -> Path:
        return self.reports_dir() / "figures"


def build_context(
    root: Path,
    config: PipelineConfig,
    config_path: Path,
    logger: logging.Logger,
    run_id: str | None = None,
) -> RunContext:
    rid = run_id or new_run_id()
    return RunContext(
        root=root.resolve(),
        run_id=rid,
        started_at=datetime.now(timezone.utc),
        config=config,
        logger=logger,
        config_path=config_path,
    )
