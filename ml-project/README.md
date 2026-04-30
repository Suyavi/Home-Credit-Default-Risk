# ML project (Home Credit-style pipeline)

## Layout

```
ml-project/
├── main.py               # python main.py — full pipeline
├── configs/
│   └── default.yaml       # Seeds, paths, training knobs, logging level
├── data/
│   ├── raw/               # Kaggle CSVs go here
│   └── processed/        # Optional wide feature exports (see config)
├── scripts/
│   ├── run_pipeline.py   # CLI: validate | run
│   └── train.py          # Shortcut: full run with default.yaml
├── src/
│   ├── pipeline/         # Orchestration, validation, manifests
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── utils.py
├── artifacts/
│   └── runs/<run_id>/    # Per-run: manifest.json, config copy, submission copy
├── models/
│   └── trained_model.pkl
├── reports/
│   ├── figures/
│   └── final_report.md
├── app/
│   └── app.py
├── requirements.txt
└── README.md
```

## Setup

```bash
cd ml-project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy competition files into `data/raw/` (`application_train.csv`, `application_test.csv`, `bureau.csv`, etc.).

## Pipeline behavior

Each **run** creates a timestamped folder under `artifacts/runs/<run_id>/` with:

- `manifest.json` — metrics (OOF AUC, fold scores, feature count), resolved config, Python version, optional git commit/dirty flag, paths to the model and submission
- `config_resolved.yaml` — copy of the YAML used for the run
- Copies of `submission.csv` and `cv_summary.md` for that run

Steps execute in order with timing logs: **validate raw, load raw, feature engineering, optional export features, matrix prep, CV training, export artifacts**.

Global and model randomness use `project.seed` from the YAML. Edit `configs/default.yaml` (or pass `--config` to a copy) to change folds, LightGBM training length, logging level, and whether processed features are written to `data/processed/`.

The `model` block sets LightGBM hyperparameters (merged onto sensible defaults) for a **bias–variance trade-off**: subsampling (`feature_fraction`, `bagging_fraction`), leaf size (`min_child_samples`), tree shape (`num_leaves`, `max_depth`), and L1/L2 (`reg_alpha`, `reg_lambda`). **Early stopping** on each validation fold limits overfitting to the training fold. After each run, check console output and `reports/cv_summary.md` for **mean train − val AUC gap** (large positive ⇒ possible overfitting) and **std of fold val AUCs** (high ⇒ unstable). If the gap is tiny but OOF AUC is low, slightly relax `model` regularization or raise `num_leaves`; if the gap is large, increase `reg_lambda` / `min_child_samples` or try a small `min_gain_to_split` (e.g. `0.01`).

## Run

**Full training** (same as `scripts/train.py`):

```bash
python scripts/run_pipeline.py run --config configs/default.yaml
```

**Check inputs only** (fast, no training):

```bash
python scripts/run_pipeline.py validate --config configs/default.yaml
```

**Shortcut** (same full pipeline):

```bash
python main.py
```

Or:

```bash
python scripts/train.py
```

Outputs: `models/trained_model.pkl`, `submission.csv` at project root, `reports/cv_summary.md`, feature-importance figure under `reports/figures/`, plus the per-run folder under `artifacts/runs/`.

## Optional API

```bash
pip install flask
python app/app.py
```

Visit `http://127.0.0.1:5000/health`.
