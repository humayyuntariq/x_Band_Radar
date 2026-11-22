"""train_gbm.py - Model 1: gradient-boosted decision trees (scikit-learn HistGradientBoosting).

Why: best default for tabular data with sharp, threshold-like decision boundaries (resolution cut-offs,
detection-range limits). Expected to be the strongest or tied-strongest model.

Run from the folder that contains scenarios/ :
  python3 train_gbm.py                       # all features, 5-fold CV
  python3 train_gbm.py --features env_only   # ablation: environment only (also: nav_only, raw)
  python3 train_gbm.py --cv 0                # skip cross-validation (faster)
Outputs: models/gbm_<features>.joblib, results/gbm_<features>.json, results/gbm_<features>_confusion.png
"""
from sklearn.ensemble import HistGradientBoostingClassifier

import radar_common as rc


def build(seed, feats):
    return HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=25, random_state=seed)


if __name__ == "__main__":
    rc.run_training("gbm", build, rc.make_parser(__doc__).parse_args())
