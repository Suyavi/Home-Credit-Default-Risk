"""Configurable training pipeline (validation → features → train → artifacts)."""

from pipeline.runner import PipelineResult, run_pipeline

__all__ = ["run_pipeline", "PipelineResult"]
