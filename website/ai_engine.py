"""Bridge between the web page and the trained mode recommender (dataset/mode_recommender.py).

    page inputs -> one scenario dict -> recommend(bundle, [scenario]) -> mode, PW, PRI

Where it looks (the folder that holds both mode_recommender.py and models/gbm_full.joblib wins):
    ../ai_model/dataset (your layout), ../ai_model, ../dataset, ./dataset, this folder, the launch folder
radar_common.py must sit next to mode_recommender.py (this folder is also on the path as a fallback).
Set the environment variable TWG_MODEL to use another model file, e.g.  TWG_MODEL=dataset/models/rf_full.joblib

Run  python3 ai_engine.py  to see what it found and to try one scenario.
"""
import importlib
import math
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent

# class index (pulse-width order) -> 3-bit code, PW, PRF. Same table as radar_common / rangeSelector.
MODES = [
    dict(idx=0, name="Mode 1", code="000", pw_ns=80,  prf_hz=3000),
    dict(idx=1, name="Mode 3", code="010", pw_ns=200, prf_hz=2000),
    dict(idx=2, name="Mode 4", code="011", pw_ns=400, prf_hz=1000),
    dict(idx=3, name="Mode 5", code="100", pw_ns=600, prf_hz=800),
    dict(idx=4, name="Mode 2", code="001", pw_ns=800, prf_hz=600),
]

# The page asks for a Douglas sea state; the model also wants wave height and wind.
# Representative values per state (assumption: middle of the Douglas wave-height band, typical wind).
WVHT_OF_STATE = {0: 0.0, 1: 0.05, 2: 0.3, 3: 0.9, 4: 1.9, 5: 3.2, 6: 5.0}
WSPD_OF_STATE = {0: 1.0, 1: 3.0, 2: 5.0, 3: 7.0, 4: 10.0, 5: 14.0, 6: 18.0}
RAIN_PATH_MAX_KM = 30.0      # same cap the training data used


class EngineUnavailable(RuntimeError):
    pass


_cache = {}


def _search_dirs():
    cwd = Path.cwd()
    seen, out = set(), []
    for d in (HERE.parent / "ai_model" / "dataset", HERE.parent / "ai_model", HERE.parent / "dataset",
              HERE / "dataset", HERE, HERE.parent, cwd / "ai_model" / "dataset", cwd / "dataset", cwd):
        d = d.resolve()
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _find():
    """(folder with mode_recommender.py, path of the model file)."""
    dirs = _search_dirs()
    has = [d for d in dirs if (d / "mode_recommender.py").is_file()]
    # prefer the folder that also holds the trained model
    root = next((d for d in has if (d / "models" / "gbm_full.joblib").is_file()), has[0] if has else None)
    if root is None:
        raise EngineUnavailable("mode_recommender.py not found. Looked in: " + ", ".join(map(str, dirs)))
    env = os.environ.get("TWG_MODEL")
    cands = [Path(env)] if env else [root / "models" / "gbm_full.joblib"] + [d / "models" / "gbm_full.joblib" for d in dirs]
    model = next((c for c in cands if c.is_file()), None)
    if model is None:
        raise EngineUnavailable("models/gbm_full.joblib not found next to mode_recommender.py "
                                f"({root}). Train it with train_gbm.py or set TWG_MODEL.")
    return root, model.resolve()


def _load():
    if "bundle" in _cache:
        return _cache["mod"], _cache["bundle"]
    root, model = _find()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        mod = importlib.import_module("mode_recommender")
        bundle = mod.load_model(str(model))
    except Exception as e:
        where = " <- ".join(f"{Path(f.filename).name}:{f.lineno} ({f.line})"
                            for f in reversed(traceback.extract_tb(e.__traceback__)[-3:]))
        raise EngineUnavailable(f"could not load the recommender from {root}: {type(e).__name__}: {e}. "
                                f"Raised at {where}")
    _cache.update(mod=mod, bundle=bundle, root=root, model=model)
    return mod, bundle


def available():
    try:
        _load()
        return True, ""
    except EngineUnavailable as e:
        return False, str(e)


def _num(body, key, default=None):
    v = body.get(key, default)
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number")
    if math.isnan(v):
        raise ValueError(f"{key} must be a number")
    return v


def build_scenario(body):
    """Page fields -> the scenario dict mode_recommender.recommend() expects (observed values)."""
    res = _num(body, "resolution_m")
    rng = _num(body, "range_km")
    if res <= 0 or rng <= 0:
        raise ValueError("Resolution and maximum range must be above 0")
    rcs_m2 = _num(body, "rcs_m2", 1.0)
    if rcs_m2 <= 0:
        raise ValueError("Target RCS must be above 0")
    sea = int(min(6, max(0, round(_num(body, "sea_state", 2)))))
    rain = max(0.0, _num(body, "rain_mmh", 0.0))
    nves = max(0, int(round(_num(body, "n_vessels", 0))))
    min_rng = _num(body, "min_range_m", 0.0)

    s = dict(
        range_req_km=rng,
        res_req_m=res,
        target_rcs_dbsm=10 * math.log10(rcs_m2),
        rain_mmph=rain,
        rain_path_km=min(RAIN_PATH_MAX_KM, rng) if rain > 0 else 0.0,
        sea_state=sea,
        wvht_m=WVHT_OF_STATE[sea],
        wspd_ms=WSPD_OF_STATE[sea],
        n_12nm=nves,
    )
    if min_rng > 0:
        s["nearest_m"] = min_rng          # closest target the operator must still see (blanking-zone check)
    return s


def _why_infeasible(s):
    """Plain-language reasons, using the same limits as the model's validator."""
    import radar_common as rc
    out = []
    if s["res_req_m"] < rc.RES_M.min():
        out.append(f"Required resolution {s['res_req_m']:g} m is finer than the best mode can resolve "
                   f"({rc.RES_M.min():g} m with the 80 ns pulse).")
    if s.get("nearest_m") is not None and s["nearest_m"] < rc.BLIND_M.min():
        out.append(f"Minimum range {s['nearest_m']:g} m is inside the shortest blind zone "
                   f"({rc.BLIND_M.min():.1f} m).")
    if s["range_req_km"] * 1e3 > rc.MAX_RANGE_M.max():
        out.append(f"Maximum range {s['range_req_km']:g} km is beyond the longest unambiguous range "
                   f"({rc.MAX_RANGE_M.max() / 1e3:g} km).")
    return out or ["No mode meets the resolution, range and minimum-range limits together."]


def synthesize(body):
    s = build_scenario(body)
    mod, bundle = _load()
    r = mod.recommend(bundle, [s])[0]

    warnings = []
    if r["warning"]:
        return {"feasible": False, "warnings": _why_infeasible(s),
                "probabilities": r.get("probabilities"), "scenario": s}

    if r.get("overridden_by_validator"):
        warnings.append(f"The model preferred {r['ml_top_choice']}, but the constraint check "
                        f"changed it to {r['mode_name']}.")
    if r.get("low_confidence"):
        warnings.append(f"Low confidence ({r['confidence']:.0%}). Check the result before relying on it.")

    m = MODES[r["mode_index"]]
    return {
        "feasible": True,
        "mode_index": m["idx"],
        "mode_name": m["name"],
        "code": r.get("mode_bits") or m["code"],
        "pw_ns": m["pw_ns"],
        "prf_hz": m["prf_hz"],
        "pri_us": round(1e6 / m["prf_hz"], 2),
        "confidence": r["confidence"],
        "low_confidence": bool(r.get("low_confidence")),
        "warnings": warnings,
        "probabilities": r.get("probabilities"),
        "scenario": s,
    }


if __name__ == "__main__":
    ok, msg = available()
    print("recommender loaded:", ok, msg)
    if ok:
        print("folder:", _cache["root"])
        print("model: ", _cache["model"])
        demo = dict(resolution_m=45, range_km=30, min_range_m=100, rcs_m2=100, sea_state=2, rain_mmh=0, n_vessels=5)
        print("try:   ", demo)
        res = synthesize(demo)
        print({k: res[k] for k in res if k not in ("scenario",)})
