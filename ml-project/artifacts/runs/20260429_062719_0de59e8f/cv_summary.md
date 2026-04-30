# Modeling summary

- **Random seed**: 42
- **OOF ROC-AUC**: 0.79418
- **Fold validation ROC-AUCs**: [0.7909879467924272, 0.8004256977504292, 0.7908193123126722, 0.7982535370032167, 0.7905660724528694]
- **Feature count**: 479

## Generalization (per-fold, at early-stopping iteration)

- **Fold train ROC-AUCs**: [0.9002768831294626, 0.9090773413621193, 0.8975547825354491, 0.882528382518816, 0.9033907385614998]
- **Mean (train − val) AUC gap**: +0.10436
  - Small gap: train and validation align (healthy).
  - Large positive gap: model fits train much better than val (risk of overfitting).
  - Large negative gap: unusual; check data or splits.
- **Std dev of fold val AUCs**: 0.00425
  - High std: unstable across folds (variance); consider more regularization or data checks.

- **Mean best boosting iteration**: 1587.2

## Run metadata

- **config**: default.yaml
- **git_commit**: unknown
- **run_id**: 20260429_062719_0de59e8f
