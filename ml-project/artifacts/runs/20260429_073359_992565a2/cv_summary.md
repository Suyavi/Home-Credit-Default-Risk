# Modeling summary

- **Random seed**: 42
- **OOF ROC-AUC**: 0.79444
- **Fold validation ROC-AUCs**: [0.7916422705943622, 0.800483298899645, 0.7912248288205812, 0.7986788606970764, 0.7902973359637704]
- **Feature count**: 495

## Generalization (per-fold, at early-stopping iteration)

- **Fold train ROC-AUCs**: [0.8867085519627184, 0.9041411668643969, 0.9100405708421779, 0.9029426194371928, 0.8994876491898548]
- **Mean (train − val) AUC gap**: +0.10620
  - Small gap: train and validation align (healthy).
  - Large positive gap: model fits train much better than val (risk of overfitting).
  - Large negative gap: unusual; check data or splits.
- **Std dev of fold val AUCs**: 0.00424
  - High std: unstable across folds (variance); consider more regularization or data checks.

- **Mean best boosting iteration**: 1607.0

## Run metadata

- **config**: default.yaml
- **git_commit**: unknown
- **run_id**: 20260429_073359_992565a2
