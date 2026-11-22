"""radar_common.py - shared physics, features, validator and metrics for the Stage 3 mode recommender.

Everything the labeler, the scenario generator, the trainers and (later) the deployed recommender must
agree on lives in this one file, so the labels and the validator can never silently disagree.

Class index = position in the pulse-width ladder (0..4). The 3-bit hardware code is a lookup only.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ======================================================================================
# 1. The five modes (ordered by pulse width) and their derived limits
# ======================================================================================
C = 3.0e8
GUARD_S = 50e-9
MODE_NAMES = ["Mode 1", "Mode 3", "Mode 4", "Mode 5", "Mode 2"]
MODE_CODES = [0b000, 0b010, 0b011, 0b100, 0b001]            # index -> 3-bit hardware code (non-monotonic)
PW_S = np.array([80e-9, 200e-9, 400e-9, 600e-9, 800e-9])
PRF_HZ = np.array([3000.0, 2000.0, 1000.0, 800.0, 600.0])
RES_M = C * PW_S / 2                                         # 12, 30, 60, 90, 120 m
MAX_RANGE_M = C / (2 * PRF_HZ)                               # 50, 75, 150, 187.5, 250 km
BLIND_M = C * (PW_S + GUARD_S) / 2                           # 19.5, 37.5, 67.5, 97.5, 127.5 m
N_MODES = 5
CODE_OF = {i: c for i, c in enumerate(MODE_CODES)}
INDEX_OF = {c: i for i, c in CODE_OF.items()}

# ======================================================================================
# 2. Labeler assumptions. EVERY constant here is an assumption: list them in the report.
# ======================================================================================
FC_GHZ = 9.41
LAMBDA_M = C / (FC_GHZ * 1e9)
RAIN_K, RAIN_ALPHA = 0.00885, 1.2739       # ITU-R P.838-3 at 9.41 GHz, circular polarisation (preprocess.py)
THRESH_DB = 13.0                           # detection threshold on SCNR (design parameter)
CAL_RANGE_KM, CAL_RCS_DBSM = 25.0, 20.0    # calibration: Mode 1 just detects 20 dBsm at 25 km, clear weather
K_DB = THRESH_DB - CAL_RCS_DBSM + 40 * np.log10(CAL_RANGE_KM)
INT_EXP = 0.8                              # non-coherent integration gain ~ N^0.8 (N proportional to PRF)
BEAM_AZ_RAD = np.radians(1.9)              # azimuth beamwidth
BEAM_EL_RAD = np.radians(20.0)             # elevation beamwidth (fan beam)
SIGMA0_REF_DB = -60.0                      # sea clutter sigma0 at sea state 0, 2 km
SIGMA0_PER_STATE_DB = 5.5                  # +5.5 dB per Douglas sea state
SIGMA0_GRAZ_SLOPE = 15.0                   # sigma0 falls 15 dB/decade of range (grazing angle)
RAIN_K2 = 0.93                             # |K|^2 of water for the rain reflectivity
FILL_M = 50_000.0                          # "no vessel / no pair within range" fill value
DOUGLAS_EDGES = [0.0, 0.1, 0.5, 1.25, 2.5, 4.0, 6.0, 9.0, 14.0]


def douglas(hs):
    return np.digitize(hs, DOUGLAS_EDGES, right=True)


def rain_loss_db(rain_mmph, path_km):
    """Two-way attenuation (dB) over the rainy part of the path."""
    r = np.maximum(np.asarray(rain_mmph, float), 0.0)
    return 2.0 * RAIN_K * r ** RAIN_ALPHA * np.asarray(path_km, float)


def sea_clutter_m2(sea_state, range_km):
    s0_db = (SIGMA0_REF_DB + SIGMA0_PER_STATE_DB * np.asarray(sea_state, float)
             - SIGMA0_GRAZ_SLOPE * np.log10(np.maximum(np.asarray(range_km, float), 2.0) / 2.0))
    area = (np.asarray(range_km, float) * 1e3)[:, None] * BEAM_AZ_RAD * RES_M[None, :]   # pulse-limited cell
    return 10 ** (s0_db[:, None] / 10) * area


def rain_clutter_m2(rain_mmph, range_km):
    r = np.maximum(np.asarray(rain_mmph, float), 0.0)
    eta = (np.pi ** 5 / LAMBDA_M ** 4) * RAIN_K2 * 200.0 * r ** 1.6 * 1e-18               # m^2/m^3, Z = 200 R^1.6
    vol = (np.pi / 4) * BEAM_AZ_RAD * BEAM_EL_RAD * ((np.asarray(range_km, float) * 1e3) ** 2)[:, None] * RES_M[None, :]
    return eta[:, None] * vol


def sinr_db(rain_mmph, path_km, sea_state, range_km, rcs_dbsm):
    """Signal-to-clutter-plus-noise (dB), shape (n, 5), one column per mode.
    Noise-limited SNR grows with pulse energy and with the number of pulses integrated (PRF);
    clutter-limited SCR falls with pulse width. Sea clutter is correlated pulse to pulse, so it is not integrated."""
    rain_mmph = np.asarray(rain_mmph, float)
    path_km = np.asarray(path_km, float)
    rain_eff = np.where(path_km > 0, rain_mmph, 0.0)
    rcs = np.asarray(rcs_dbsm, float)
    snr_db = ((K_DB + rcs - 40 * np.log10(np.asarray(range_km, float)) - rain_loss_db(rain_eff, path_km))[:, None]
              + 10 * np.log10(PW_S / PW_S[0])[None, :]
              + 10 * INT_EXP * np.log10(PRF_HZ / PRF_HZ[0])[None, :])
    clutter = sea_clutter_m2(sea_state, range_km) + rain_clutter_m2(rain_eff, range_km)
    scr = (10 ** (rcs / 10))[:, None] / clutter
    return 10 * np.log10(1.0 / (1.0 / 10 ** (snr_db / 10) + 1.0 / scr))


def best_clear_gain_db():
    """Largest noise-limited gain over Mode 1 in clear weather (pulse energy minus the integration lost to a lower PRF)."""
    return float(np.max(10 * np.log10(PW_S / PW_S[0]) + 10 * INT_EXP * np.log10(PRF_HZ / PRF_HZ[0])))


def feasibility(range_km, res_req_m, minpair_m, nearest_m):
    """(n, 5) bool. The SAME function is used by the labeler (true values) and the validator (observed values)."""
    R = np.asarray(range_km, float) * 1e3
    res_eff = np.minimum(np.asarray(res_req_m, float), np.asarray(minpair_m, float))
    closest = np.minimum(np.asarray(nearest_m, float), R)
    return ((RES_M[None, :] <= res_eff[:, None] + 1e-9)             # mode can resolve the required separation
            & (R[:, None] <= MAX_RANGE_M[None, :])                  # no range ambiguity
            & (closest[:, None] >= BLIND_M[None, :]))               # closest target is outside the blanking zone


def label_scenarios(d):
    """d holds the TRUE (noise-free) scenario values. Returns dict with label (-1 = no feasible mode),
    feasibility matrix, SCNR matrix and the decision margin (dB) used to place label noise.
    Rule: shortest feasible mode whose SCNR >= THRESH_DB; if none reaches it, the feasible mode with the best SCNR."""
    feas = feasibility(d["range_req_km"], d["res_req_m"], d["minpair_m_true"], d["nearest_m_true"])
    sinr = sinr_db(d["rain_mmph_true"], d["rain_path_km_true"], d["sea_state_true"], d["range_req_km"], d["rcs_dbsm_true"])
    n = len(feas)
    reach = feas & (sinr >= THRESH_DB)
    any_f, any_r = feas.any(1), reach.any(1)
    masked = np.where(feas, sinr, -np.inf)
    label = np.where(any_r, reach.argmax(1), masked.argmax(1))
    label[~any_f] = -1
    lab = np.clip(label, 0, N_MODES - 1)
    ar = np.arange(n)
    before = feas & (np.arange(N_MODES)[None, :] < lab[:, None])
    max_before = np.where(before, sinr, -np.inf).max(1)
    m_thr = np.minimum(sinr[ar, lab] - THRESH_DB, THRESH_DB - max_before)
    srt = -np.sort(-masked, axis=1)
    with np.errstate(invalid="ignore"):
        gap = srt[:, 0] - srt[:, 1]
    margin = np.where(any_r, m_thr, gap)
    margin = np.where(np.isnan(margin) | ~any_f, np.inf, margin)
    return dict(label=label, feas=feas, sinr=sinr, margin=margin, rule_threshold=any_r & any_f)


# ======================================================================================
# 3. Sensor noise (applied AFTER labelling) and engineered features
# ======================================================================================
def add_sensor_noise(d, rng):
    """Turn true values into what the deployed system would actually observe."""
    n = len(d)
    o = d.copy()
    o["rain_mmph"] = d["rain_mmph_true"] * np.exp(rng.normal(0, 0.25, n))
    o["rain_path_km"] = np.minimum(d["rain_path_km_true"] * np.exp(rng.normal(0, 0.20, n)), d["range_req_km"])
    o["wvht_m"] = np.maximum(d["wvht_m_true"] + rng.normal(0, 0.15, n), 0.0)
    o["wspd_ms"] = np.maximum(d["wspd_ms_true"] + rng.normal(0, 1.0, n), 0.0)
    o["sea_state"] = np.clip(douglas(o["wvht_m"].to_numpy()), 0, 6)
    o["n_12nm"] = np.maximum(np.round(d["n_12nm_true"] * np.exp(rng.normal(0, 0.15, n))), 0)
    for col in ("minpair_m", "nearest_m"):
        t = d[col + "_true"].to_numpy()
        o[col] = np.where(t >= FILL_M, FILL_M, t * np.exp(rng.normal(0, 0.10, n)))
    o["target_rcs_dbsm"] = d["rcs_dbsm_true"] + rng.normal(0, 2.0, n)
    return o


def add_engineered(d):
    d["rain_loss_db"] = rain_loss_db(d["rain_mmph"], d["rain_path_km"])
    d["res_eff_m"] = np.minimum(d["res_req_m"], d["minpair_m"])
    d["closest_eff_m"] = np.minimum(d["nearest_m"], d["range_req_km"] * 1e3)
    d["log10_range_km"] = np.log10(d["range_req_km"])
    return d


ENV = ["rain_mmph", "sea_state", "wvht_m", "wspd_ms", "n_12nm", "minpair_m", "nearest_m"]   # exist without any mission
NAV = ["range_req_km", "res_req_m", "target_rcs_dbsm"]                                       # what the operator needs
PATH = ["rain_path_km"]                          # depends on the mission range, so it is kept out of env_only
ENG = ["rain_loss_db", "res_eff_m", "closest_eff_m", "log10_range_km"]
FEATURE_SETS = {"full": ENV + NAV + PATH + ENG, "raw": ENV + NAV + PATH, "env_only": ENV, "nav_only": NAV}
HEAVY_TAIL = ["n_12nm", "minpair_m", "nearest_m", "range_req_km", "res_req_m", "rain_mmph", "rain_path_km",
              "rain_loss_db", "res_eff_m", "closest_eff_m"]


# ======================================================================================
# 4. Validator: highest-probability mode that is feasible; shares feasibility() with the labeler
# ======================================================================================
def validate(proba, d_obs):
    """proba (n,5), d_obs has the OBSERVED features. Returns (chosen index or -1, overridden flag)."""
    n = len(proba)
    feas = feasibility(d_obs["range_req_km"], d_obs["res_req_m"], d_obs["minpair_m"], d_obs["nearest_m"])
    order = np.argsort(-proba, axis=1)
    chosen = np.full(n, -1)
    ar = np.arange(n)
    for rank in range(N_MODES):
        cand = order[:, rank]
        ok = feas[ar, cand] & (chosen < 0)
        chosen[ok] = cand[ok]
    return chosen, (chosen >= 0) & (chosen != order[:, 0])


# ======================================================================================
# 5. Metrics
# ======================================================================================
def _macro_f1(y, p):
    f = []
    for k in range(N_MODES):
        tp = np.sum((y == k) & (p == k)); fp = np.sum((y != k) & (p == k)); fn = np.sum((y == k) & (p != k))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    present = [f[k] for k in range(N_MODES) if np.any(y == k)]
    return float(np.mean(present)) if present else float("nan")


def ordinal_metrics(y, p):
    return {"accuracy": float(np.mean(y == p)), "macro_f1": _macro_f1(y, p),
            "adjacent_acc": float(np.mean(np.abs(y - p) <= 1)), "ordinal_mae": float(np.mean(np.abs(y - p))),
            "gross_error": float(np.mean(np.abs(y - p) >= 2))}


def full_proba(model, X):
    p = model.predict_proba(X)
    out = np.zeros((len(X), N_MODES))
    out[:, np.asarray(model.classes_, int)] = p
    return out


def evaluate_set(predict_proba, d, feats):
    """predict_proba: callable taking a feature frame. d must contain the true columns for feasibility."""
    n = len(d)
    t0 = time.perf_counter()
    proba = predict_proba(d[feats])
    ms_per_1k = (time.perf_counter() - t0) / max(n, 1) * 1e6
    pred = proba.argmax(1)
    ar = np.arange(n)
    feas_true = feasibility(d["range_req_km"], d["res_req_m"], d["minpair_m_true"], d["nearest_m_true"])
    chosen, over = validate(proba, d)
    has = chosen >= 0
    ch = np.clip(chosen, 0, N_MODES - 1)
    res = {"n": int(n), "ms_per_1000": float(ms_per_1k)}
    if "label" in d:
        y, yc = d["label"].to_numpy(int), d["label_clean"].to_numpy(int)
        ok = y >= 0
        if ok.any():
            res["vs_noisy"] = ordinal_metrics(y[ok], pred[ok])
            res["vs_clean"] = ordinal_metrics(yc[ok], pred[ok])
            res["validated_vs_clean"] = ordinal_metrics(yc[ok & has], chosen[ok & has]) if (ok & has).any() else {}
            res["confusion_vs_clean"] = [[int(np.sum((yc[ok] == a) & (pred[ok] == b))) for b in range(N_MODES)]
                                         for a in range(N_MODES)]
            res["violation_raw"] = float(np.mean(~feas_true[ar, pred][ok]))
            res["violation_after_validator"] = float(np.mean((has & ~feas_true[ar, ch])[ok]))
            res["override_rate"] = float(np.mean(over[ok]))
            res["false_warning_rate"] = float(np.mean(~has[ok]))
        else:                                                    # infeasible set: label = -1 for every row
            res["warning_rate"] = float(np.mean(~has))
            res["violation_raw"] = float(np.mean(~feas_true[ar, pred]))
    return res


# ======================================================================================
# 6. Data loading and the shared training driver used by train_*.py
# ======================================================================================
def load_splits(data_dir, seed=42):
    from sklearn.model_selection import train_test_split
    d = Path(data_dir)
    bal = pd.read_csv(d / "dataset.csv")
    nat = pd.read_csv(d / "dataset_natural.csv")
    inf = pd.read_csv(d / "dataset_infeasible.csv")
    tr, tmp = train_test_split(bal, test_size=0.30, stratify=bal["label"], random_state=seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=tmp["label"], random_state=seed)
    return dict(train=tr, val=va, test=te, natural=nat, infeasible=inf)


def eval_sets(splits):
    nat = splits["natural"]
    return {"val": splits["val"], "test": splits["test"], "natural": nat,
            "natural_edge": nat[nat["edge"] == 1], "natural_real": nat[nat["source"] == "real"],
            "natural_stress": nat[nat["source"] == "stress"], "infeasible": splits["infeasible"]}


def make_parser(description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--data-dir", default="scenarios")
    ap.add_argument("--features", default="full", choices=list(FEATURE_SETS))
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv", type=int, default=5, help="folds of stratified cross-validation (0 = skip)")
    return ap


def run_training(model_name, build_model, args):
    import joblib
    from sklearn.base import clone
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import StratifiedKFold

    feats = FEATURE_SETS[args.features]
    splits = load_splits(args.data_dir, args.seed)
    tr, va = splits["train"], splits["val"]
    print(f"[{model_name}] features={args.features} ({len(feats)})  train={len(tr)} val={len(va)} "
          f"test={len(splits['test'])} natural={len(splits['natural'])}")
    model = build_model(args.seed, feats)
    t0 = time.perf_counter()
    model.fit(tr[feats], tr["label"])
    fit_s = time.perf_counter() - t0
    print(f"[{model_name}] fitted in {fit_s:.1f} s")

    result = {"model": model_name, "features": args.features, "n_features": len(feats), "fit_seconds": fit_s,
              "n_train": int(len(tr)), "sets": {}}
    if args.cv and args.cv > 1:
        full = pd.concat([tr, va], ignore_index=True)
        f1s = []
        for k, (a, b) in enumerate(StratifiedKFold(args.cv, shuffle=True, random_state=args.seed).split(full, full["label"])):
            m = clone(model)
            m.fit(full.iloc[a][feats], full.iloc[a]["label"])
            p = full_proba(m, full.iloc[b][feats]).argmax(1)
            f1s.append(_macro_f1(full.iloc[b]["label_clean"].to_numpy(int), p))
        result["cv_macro_f1_vs_clean"] = {"mean": float(np.mean(f1s)), "std": float(np.std(f1s)), "folds": f1s}
        print(f"[{model_name}] {args.cv}-fold CV macro-F1 vs clean labels: {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")

    for name, d in eval_sets(splits).items():
        if len(d):
            result["sets"][name] = evaluate_set(lambda X: full_proba(model, X), d, feats)

    pi = permutation_importance(model, va[feats], va["label"], scoring="f1_macro", n_repeats=3,
                                random_state=args.seed, n_jobs=1)
    result["importance"] = dict(sorted(zip(feats, map(float, pi.importances_mean)), key=lambda kv: -kv[1]))

    tag = f"{model_name}_{args.features}"
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    mp = Path(args.models_dir) / f"{tag}.joblib"
    joblib.dump({"model": model, "features": feats, "model_name": model_name}, mp, compress=3)
    result["model_kb"] = mp.stat().st_size / 1024
    (Path(args.results_dir) / f"{tag}.json").write_text(json.dumps(result, indent=2))
    _confusion_plot(result, Path(args.results_dir) / f"{tag}_confusion.png")
    _print_summary(result)
    return result


def _confusion_plot(result, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [n for n in ("test", "natural") if n in result["sets"]]
        fig, ax = plt.subplots(1, len(names), figsize=(5.5 * len(names), 4.8))
        ax = np.atleast_1d(ax)
        for a, nm in zip(ax, names):
            cm = np.array(result["sets"][nm]["confusion_vs_clean"], float)
            cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            a.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
            for i in range(N_MODES):
                for j in range(N_MODES):
                    a.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8,
                           color="white" if cmn[i, j] > 0.5 else "black")
            a.set_xticks(range(N_MODES), MODE_NAMES, rotation=45)
            a.set_yticks(range(N_MODES), MODE_NAMES)
            a.set(title=f"{result['model']} / {result['features']} - {nm}", xlabel="predicted", ylabel="true (clean)")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
    except Exception as e:
        print("confusion plot skipped:", e)


def _print_summary(r):
    print(f"\n{'set':16s}{'n':>7s}{'acc':>7s}{'F1':>7s}{'adj':>7s}{'MAE':>7s} | {'viol raw':>9s}{'viol val':>9s}{'override':>9s}")
    for nm, s in r["sets"].items():
        if "vs_clean" in s:
            m = s["vs_clean"]
            print(f"{nm:16s}{s['n']:7d}{m['accuracy']:7.3f}{m['macro_f1']:7.3f}{m['adjacent_acc']:7.3f}{m['ordinal_mae']:7.3f}"
                  f" | {s['violation_raw']:9.3f}{s['violation_after_validator']:9.3f}{s['override_rate']:9.3f}")
        else:
            print(f"{nm:16s}{s['n']:7d}   validator raised a 'no feasible mode' warning on {s['warning_rate']:.1%} of rows")
    top = list(r["importance"].items())[:6]
    print("top features (permutation, macro-F1 drop):", ", ".join(f"{k} {v:.3f}" for k, v in top))
    print("(accuracy columns are measured against the CLEAN labels; noisy-label scores are in the JSON)\n")
