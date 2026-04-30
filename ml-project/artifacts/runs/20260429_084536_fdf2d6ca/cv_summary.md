# Modeling summary

- **Random seed**: 42
- **OOF ROC-AUC**: 0.79631
- **Fold validation ROC-AUCs**: [0.7937316411978429, 0.7953418972931741, 0.7901285842018823, 0.7967558679367276, 0.7946341988889533]
- **Feature count**: 491

## Generalization (per-fold, at early-stopping iteration)

- **Fold train ROC-AUCs**: [0.8831755311126805, 0.9010801796047238, 0.8902066690663477, 0.8911058011195645, 0.8999420349956843]
- **Mean (train − val) AUC gap**: +0.09898
  - Small gap: train and validation align (healthy).
  - Large positive gap: model fits train much better than val (risk of overfitting).
  - Large negative gap: unusual; check data or splits.
- **Std dev of fold val AUCs**: 0.00223
  - High std: unstable across folds (variance); consider more regularization or data checks.

- **Mean best boosting iteration**: 1455.9

## Run metadata

- **config**: default.yaml
- **ensemble_seeds**: 42,119,2069
- **git_commit**: unknown
- **run_id**: 20260429_084536_fdf2d6ca
