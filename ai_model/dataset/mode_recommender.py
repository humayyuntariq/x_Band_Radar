"""mode_recommender.py - scenario in, validated 3-bit TWG mode out.

  scenario  ->  engineered features  ->  trained model (probabilities for the 5 modes)
            ->  validator (same feasibility function as the labeler; falls back to the next most likely feasible mode)
            ->  mode index, name and the 3-bit code for the VHDL/MyHDL `mode` input.

Use from Python (e.g. in testbench.py):
    from mode_recommender import load_model, recommend
    bundle = load_model("models/gbm_full.joblib")
    r = recommend(bundle, dict(range_req_km=12, res_req_m=70, target_rcs_dbsm=18, ...))[0]
    mode_bits = r["mode_code"]        # 0b000, 0b010, 0b011, 0b100 or 0b001; None if no mode is feasible

Command line:
    python3 mode_recommender.py --demo                   # six named maritime scenarios
    python3 mode_recommender.py --model models/rf_full.joblib --demo

Inputs per scenario (OBSERVED values, i.e. what the sensors report):
  rain_mmph, rain_path_km (length of the rainy part of the path), sea_state (Douglas 0-6; derived from wvht_m if missing),
  wvht_m, wspd_ms, n_12nm (vessels within 12 NM), minpair_m (closest pair of vessels, omit if none),
  nearest_m (nearest vessel, omit if none), range_req_km, res_req_m (smallest separation to resolve), target_rcs_dbsm.
"""
import argparse

import joblib
import numpy as np
import pandas as pd

import radar_common as rc

REQUIRED = ["rain_mmph", "rain_path_km", "wvht_m", "wspd_ms", "n_12nm", "range_req_km", "res_req_m", "target_rcs_dbsm"]


def load_model(path="models/gbm_full.joblib"):
    return joblib.load(path)


def _frame(scenarios):
    d = pd.DataFrame([scenarios] if isinstance(scenarios, dict) else scenarios).copy()
    missing = [c for c in REQUIRED if c not in d.columns]
    if missing:
        raise ValueError(f"scenario is missing: {missing}")
    for c in ("minpair_m", "nearest_m"):                       # "no vessel / no pair" is encoded as FILL_M
        if c not in d.columns:
            d[c] = rc.FILL_M
        d[c] = d[c].fillna(rc.FILL_M)
    if "sea_state" not in d.columns:
        d["sea_state"] = np.clip(rc.douglas(d["wvht_m"].to_numpy(float)), 0, 6)
    return rc.add_engineered(d)


def recommend(bundle, scenarios):
    """Returns one dict per scenario. mode_index/mode_code are None when no mode can satisfy the constraints."""
    d = _frame(scenarios)
    proba = rc.full_proba(bundle["model"], d[bundle["features"]])
    chosen, overridden = rc.validate(proba, d)
    out = []
    for i in range(len(d)):
        top = int(proba[i].argmax())
        k = int(chosen[i])
        out.append({
            "mode_index": k if k >= 0 else None,
            "mode_name": rc.MODE_NAMES[k] if k >= 0 else "NO FEASIBLE MODE",
            "mode_code": rc.CODE_OF[k] if k >= 0 else None,
            "mode_bits": format(rc.CODE_OF[k], "03b") if k >= 0 else None,
            "ml_top_choice": rc.MODE_NAMES[top],
            "overridden_by_validator": bool(overridden[i]),
            "warning": k < 0,
            "confidence": float(proba[i, k]) if k >= 0 else 0.0,
            "probabilities": {rc.MODE_NAMES[j]: round(float(proba[i, j]), 3) for j in range(rc.N_MODES)},
        })
    return out


_CALM = dict(rain_mmph=0, rain_path_km=0, sea_state=3, wvht_m=1.0, wspd_ms=6, n_12nm=1)
DEMO = {
    "1 Harbour docking": dict(range_req_km=0.5, res_req_m=15, target_rcs_dbsm=10, rain_mmph=0, rain_path_km=0,
                              sea_state=1, wvht_m=0.2, wspd_ms=3, n_12nm=40, minpair_m=25, nearest_m=40),
    "2 Fairway transit": dict(range_req_km=4, res_req_m=35, target_rcs_dbsm=20, rain_mmph=0, rain_path_km=0,
                              sea_state=2, wvht_m=0.4, wspd_ms=5, n_12nm=15, minpair_m=60, nearest_m=300),
    "3 Approach, far target": dict(_CALM, range_req_km=30, res_req_m=45, target_rcs_dbsm=20),
    "4 Offshore lane, 70 m": dict(_CALM, range_req_km=40, res_req_m=70, target_rcs_dbsm=20),
    "5 Offshore, 100 m": dict(_CALM, range_req_km=50, res_req_m=100, target_rcs_dbsm=25),
    "6 Long-range surveillance": dict(_CALM, range_req_km=80, res_req_m=150, target_rcs_dbsm=25),
    "7 Offshore squall": dict(range_req_km=30, res_req_m=150, target_rcs_dbsm=25, rain_mmph=20, rain_path_km=20,
                              sea_state=5, wvht_m=3.0, wspd_ms=16, n_12nm=3),
    "8 Impossible request": dict(range_req_km=5, res_req_m=8, target_rcs_dbsm=20, rain_mmph=0, rain_path_km=0,
                                 sea_state=2, wvht_m=0.4, wspd_ms=5, n_12nm=5, minpair_m=200, nearest_m=500),
}


def physics_labels(scenarios):
    """What the radar-equation labeler says for the same inputs (observed values treated as true). For the demo only."""
    d = _frame(scenarios)
    t = pd.DataFrame({"rain_mmph_true": d["rain_mmph"], "rain_path_km_true": d["rain_path_km"],
                      "sea_state_true": d["sea_state"], "range_req_km": d["range_req_km"], "res_req_m": d["res_req_m"],
                      "minpair_m_true": d["minpair_m"], "nearest_m_true": d["nearest_m"],
                      "rcs_dbsm_true": d["target_rcs_dbsm"]})
    return rc.label_scenarios(t)["label"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/gbm_full.joblib")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    bundle = load_model(args.model)
    if not args.demo:
        print("Nothing to do. Use --demo, or import recommend() from your own code.")
        return
    res = recommend(bundle, list(DEMO.values()))
    phys = physics_labels(list(DEMO.values()))
    print(f"model: {bundle['model_name']}   features: {len(bundle['features'])}\n")
    print(f"{'scenario':28s}{'ML + validator':>16s}{'bits':>6s}{'conf':>7s}{'physics label':>15s}  note")
    for (name, _), r, p in zip(DEMO.items(), res, phys):
        note = ("WARNING: no mode satisfies the constraints" if r["warning"]
                else f"validator changed {r['ml_top_choice']} -> {r['mode_name']}" if r["overridden_by_validator"] else "")
        print(f"{name:28s}{r['mode_name']:>16s}{(r['mode_bits'] or '-'):>6s}{r['confidence']:7.2f}"
              f"{(rc.MODE_NAMES[p] if p >= 0 else 'none'):>15s}  {note}")


if __name__ == "__main__":
    main()
