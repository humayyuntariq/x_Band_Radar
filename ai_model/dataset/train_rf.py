"""train_rf.py - Model 2: Random Forest.

Why: strong, needs no feature scaling, and averages over many trees, which makes it robust to the sensor noise
and boundary label noise in the dataset. A second tree-based model shows whether boosting really helps.

Run from the folder that contains scenarios/ :
  python3 train_rf.py
  python3 train_rf.py --features env_only    # ablation (also: nav_only, raw)
  python3 train_rf.py --cv 0
Outputs: models/rf_<features>.joblib, results/rf_<features>.json, results/rf_<features>_confusion.png
"""
from sklearn.ensemble import RandomForestClassifier

import radar_common as rc


def build(seed, feats):
    # min_samples_leaf=5 keeps the saved forest to a few tens of MB (leaf=2 made it ~170 MB)
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=5, max_features="sqrt",
                                  n_jobs=-1, random_state=seed)


if __name__ == "__main__":
    rc.run_training("rf", build, rc.make_parser(__doc__).parse_args())
