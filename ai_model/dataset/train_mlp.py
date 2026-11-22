"""train_mlp.py - Model 3: multilayer perceptron (neural network).

Why: a smooth, non-tree model as a fair comparison. Heavy-tailed inputs (range, vessel counts, spacing, rain)
are log-transformed and every input is standardised. It is expected to be competitive but not better than the
tree models on ~20 tabular features; smooth activations blur hard cut-offs such as "resolution <= separation".

Run from the folder that contains scenarios/ :
  python3 train_mlp.py
  python3 train_mlp.py --features env_only   # ablation (also: nav_only, raw)
  python3 train_mlp.py --cv 0
Outputs: models/mlp_<features>.joblib, results/mlp_<features>.json, results/mlp_<features>_confusion.png
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import radar_common as rc


def build(seed, feats):
    log_cols = [c for c in feats if c in rc.HEAVY_TAIL]
    lin_cols = [c for c in feats if c not in rc.HEAVY_TAIL]
    parts = []
    if log_cols:
        parts.append(("log", Pipeline([("log1p", FunctionTransformer(np.log1p)), ("scale", StandardScaler())]), log_cols))
    if lin_cols:
        parts.append(("lin", StandardScaler(), lin_cols))
    return Pipeline([
        ("prep", ColumnTransformer(parts)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64, 32), alpha=1e-4, learning_rate_init=1e-3,
                              early_stopping=True, validation_fraction=0.1, n_iter_no_change=15,
                              max_iter=300, random_state=seed)),
    ])


if __name__ == "__main__":
    rc.run_training("mlp", build, rc.make_parser(__doc__).parse_args())
