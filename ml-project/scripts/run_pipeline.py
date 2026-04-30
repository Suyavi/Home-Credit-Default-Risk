#!/usr/bin/env python3
"""CLI for the ML pipeline: validate raw data or run full training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.runner import run_pipeline, validate_only


def main() -> None:
    parser = argparse.ArgumentParser(description="Home Credit-style ML pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate", help="Check that raw CSVs exist (and schema if strict).")
    p_val.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "default.yaml",
        help="Path to pipeline YAML.",
    )

    p_run = sub.add_parser("run", help="Full pipeline: validate, features, train, artifacts.")
    p_run.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "default.yaml",
        help="Path to pipeline YAML.",
    )
    p_run.add_argument("--run-id", type=str, default=None, help="Optional fixed run id for the artifact folder.")

    args = parser.parse_args()
    cfg_path: Path = args.config
    if not cfg_path.is_file():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    if args.command == "validate":
        validate_only(project_root=ROOT, config_path=cfg_path)
        return

    if args.command == "run":
        run_id = getattr(args, "run_id", None)
        run_pipeline(project_root=ROOT, config_path=cfg_path, run_id=run_id)
        return


if __name__ == "__main__":
    main()
