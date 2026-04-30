# Modeling summary

- **Random seed**: 42
- **OOF ROC-AUC**: 0.79650
- **Fold validation ROC-AUCs**: [0.7941465516554495, 0.7954609992797748, 0.7895595812236023, 0.7966603766382719, 0.7950477617692847]
- **Feature count**: 499

## Generalization (per-fold, at early-stopping iteration)

- **Fold train ROC-AUCs**: [0.8980329366219982, 0.8969761663385158, 0.8928738447801923, 0.8981410583973776, 0.9021703462530312]
- **Mean (train − val) AUC gap**: +0.10346
  - Small gap: train and validation align (healthy).
  - Large positive gap: model fits train much better than val (risk of overfitting).
  - Large negative gap: unusual; check data or splits.
- **Std dev of fold val AUCs**: 0.00245
  - High std: unstable across folds (variance); consider more regularization or data checks.

- **Mean best boosting iteration**: 1563.2

## Run metadata

- **config**: default.yaml
- **ensemble_seeds**: 42,119,2069
- **git_commit**: unknown
- **run_id**: 20260429_101858_a718fb60
