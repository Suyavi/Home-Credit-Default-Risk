#!/usr/bin/env python3
"""Entry point: run the full ML pipeline (same as `python scripts/run_pipeline.py run`)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.runner import run_pipeline


def main() -> None:
    config_path = ROOT / "configs" / "default.yaml"
    if not config_path.is_file():
        print(f"Missing config: {config_path}", file=sys.stderr)
        sys.exit(1)
    run_pipeline(project_root=ROOT, config_path=config_path)


if __name__ == "__main__":
    main()
