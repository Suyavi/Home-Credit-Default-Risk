# Final report

## Executive summary

_(Fill after experiments: business question, best model, key metrics.)_

## Data

- Raw inputs: `data/raw/`
- Engineered tables: `data/processed/` (optional CSV exports)

## Modeling

- Training code: `src/model_training.py`
- Cross-validation notes: see `reports/cv_summary.md` after running `python scripts/train.py`

## Artifacts

- Trained booster: `models/trained_model.pkl`
- Kaggle-style submission: `submission.csv` (project root, written by `scripts/train.py`)
- Figures: `reports/figures/`

## Next steps

- [ ] Tune hyperparameters (Optuna)
- [ ] Add leakage-safe target encoding if needed
- [ ] Wire `app/app.py` to load `trained_model.pkl` and expose `/predict`
