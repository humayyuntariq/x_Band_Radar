"""compare_models.py - comparative performance analysis over everything in results/.

Run after the train_*.py scripts (also after the --features ablation runs):
  python3 compare_models.py
Prints and saves results/comparison.md and results/comparison.csv.

Baselines are evaluated on exactly the same data as the models:
  always_mode1     always picks Mode 1 (the majority class of the natural mix)
  range_scale_rule operator-style rule on required range only. The thresholds (2.8, 7.4, 18.5, 33.3 km = 1.5, 4, 10, 18 NM)
                   are an ASSUMPTION, not derived from the physics.
  coarsest_feasible picks the longest-pulse mode that satisfies the constraints (maximum energy, no weather logic)
All accuracy numbers are measured against the CLEAN labels (the noisy-label scores are in each JSON).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import radar_common as rc

RANGE_EDGES_KM = [2.8, 7.4, 18.5, 33.3]


def proba_from_pred(pred):
    return 1.0 / (1.0 + np.abs(np.arange(rc.N_MODES)[None, :] - np.asarray(pred)[:, None]))


def baselines():
    def always1(X):
        return proba_from_pred(np.zeros(len(X), int))

    def range_scale(X):
        return proba_from_pred(np.digitize(X["range_req_km"].to_numpy(), RANGE_EDGES_KM))

    def coarsest(X):
        feas = rc.feasibility(X["range_req_km"], X["res_req_m"], X["minpair_m"], X["nearest_m"])
        pred = np.where(feas.any(1), (feas * np.arange(rc.N_MODES)[None, :]).max(1), 0)
        return proba_from_pred(pred)

    cols = ["range_req_km", "res_req_m", "minpair_m", "nearest_m"]
    return {"always_mode1": (always1, cols), "range_scale_rule": (range_scale, cols), "coarsest_feasible": (coarsest, cols)}


def row(name, feats, r):
    g = lambda s, k, m="vs_clean": r["sets"].get(s, {}).get(m, {}).get(k, np.nan)
    return {"model": name, "features": feats,
            "cv_F1": r.get("cv_macro_f1_vs_clean", {}).get("mean", np.nan),
            "test_acc": g("test", "accuracy"), "test_F1": g("test", "macro_f1"),
            "test_adj": g("test", "adjacent_acc"), "test_MAE": g("test", "ordinal_mae"),
            "nat_acc": g("natural", "accuracy"), "nat_F1": g("natural", "macro_f1"),
            "edge_F1": g("natural_edge", "macro_f1"),
            "real_F1": g("natural_real", "macro_f1"), "stress_F1": g("natural_stress", "macro_f1"),
            "viol_raw": r["sets"]["natural"].get("violation_raw", np.nan),
            "viol_after": r["sets"]["natural"].get("violation_after_validator", np.nan),
            "warn_infeas": r["sets"].get("infeasible", {}).get("warning_rate", np.nan),
            "ms_per_1k": r["sets"]["test"]["ms_per_1000"], "model_KB": r.get("model_kb", np.nan),
            "fit_s": r.get("fit_seconds", np.nan)}


def md(df, floatfmt=3):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = [f"{v:.{floatfmt}f}" if isinstance(v, (float, np.floating)) and not np.isnan(v)
                 else ("" if isinstance(v, (float, np.floating)) else str(v)) for v in r]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--data-dir", default="scenarios")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rd = Path(args.results_dir)
    files = sorted(p for p in rd.glob("*.json") if p.stem not in ("comparison",))
    if not files:
        raise SystemExit(f"No result files in {rd}/. Run train_gbm.py, train_rf.py, train_mlp.py first.")
    rows = []
    for p in files:
        r = json.loads(p.read_text())
        rows.append(row(r["model"], r["features"], r))

    splits = rc.load_splits(args.data_dir, args.seed)
    sets = rc.eval_sets(splits)
    for name, (fn, cols) in baselines().items():
        res = {"sets": {s: rc.evaluate_set(fn, d, cols) for s, d in sets.items() if len(d)}, "model_kb": 0.0}
        rows.append(row(name, "rule", res))

    df = pd.DataFrame(rows)
    df.to_csv(rd / "comparison.csv", index=False)

    full = df[df.features.isin(["full", "rule"])].sort_values("test_F1", ascending=False)
    t1 = full[["model", "cv_F1", "test_acc", "test_F1", "test_adj", "test_MAE", "nat_acc", "nat_F1", "edge_F1"]]
    t2 = full[["model", "viol_raw", "viol_after", "warn_infeas", "ms_per_1k", "model_KB", "fit_s"]]
    abl = df[df.model.isin(["gbm", "rf", "mlp"])].pivot_table(index="model", columns="features",
                                                              values=["test_F1", "edge_F1"])
    abl.columns = [f"{a}:{b}" for a, b in abl.columns]
    abl = abl.reset_index()

    text = ("## 1. Model comparison (all features)\n" + md(t1)
            + "\n\n## 2. Constraint safety and cost (natural mix)\n" + md(t2, 4)
            + "\n\n## 3. Ablation: which inputs does the classifier need? (macro-F1)\n" + md(abl)
            + "\n\nfeature sets: full = env + nav + engineered, raw = env + nav, env_only = environment only, "
              "nav_only = range/resolution/RCS only")
    (rd / "comparison.md").write_text(text)
    print(text)
    print(f"\nSaved {rd / 'comparison.md'} and {rd / 'comparison.csv'}")


if __name__ == "__main__":
    main()
