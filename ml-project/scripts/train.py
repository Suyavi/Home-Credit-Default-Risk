"""Train LightGBM with CV (delegates to the configurable pipeline)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.runner import run_pipeline


def main() -> None:
    cfg = ROOT / "configs" / "default.yaml"
    run_pipeline(project_root=ROOT, config_path=cfg)


if __name__ == "__main__":
    main()
