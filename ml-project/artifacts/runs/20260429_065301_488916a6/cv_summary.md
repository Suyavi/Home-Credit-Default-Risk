# Modeling summary

- **Random seed**: 42
- **OOF ROC-AUC**: 0.79439
- **Fold validation ROC-AUCs**: [0.7907032306551962, 0.8003950856732932, 0.7914816631858922, 0.7985892153684311, 0.7909819414216369]
- **Feature count**: 492

## Generalization (per-fold, at early-stopping iteration)

- **Fold train ROC-AUCs**: [0.9010275859416788, 0.8934378286019551, 0.9009802892786639, 0.9035657528447201, 0.9101140681799779]
- **Mean (train − val) AUC gap**: +0.10739
  - Small gap: train and validation align (healthy).
  - Large positive gap: model fits train much better than val (risk of overfitting).
  - Large negative gap: unusual; check data or splits.
- **Std dev of fold val AUCs**: 0.00418
  - High std: unstable across folds (variance); consider more regularization or data checks.

- **Mean best boosting iteration**: 1636.6

## Run metadata

- **config**: default.yaml
- **git_commit**: unknown
- **run_id**: 20260429_065301_488916a6
