# Modeling summary

- **Random seed**: 42
- **OOF ROC-AUC**: 0.79371
- **Fold validation ROC-AUCs**: [0.7902918540790521, 0.8000602907940706, 0.7900291623407469, 0.7976763000304892, 0.7905651569048857]
- **Feature count**: 448

## Generalization (per-fold, at early-stopping iteration)

- **Fold train ROC-AUCs**: [0.8864000744123202, 0.8938369360677405, 0.8879092386207053, 0.8851665050014683, 0.8941356681235526]
- **Mean (train − val) AUC gap**: +0.09577
  - Small gap: train and validation align (healthy).
  - Large positive gap: model fits train much better than val (risk of overfitting).
  - Large negative gap: unusual; check data or splits.
- **Std dev of fold val AUCs**: 0.00427
  - High std: unstable across folds (variance); consider more regularization or data checks.

- **Mean best boosting iteration**: 1383.4

## Run metadata

- **config**: default.yaml
- **git_commit**: unknown
- **run_id**: 20260429_060255_e88b4107
