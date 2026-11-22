"""scenario_gen.py - build the labelled scenario dataset for the radar mode recommender.

Inputs  (from preprocess.py):  processed/hourly_sea_rain.csv, processed/ais_traffic.csv
Outputs (in ./scenarios/):
  dataset.csv             class-balanced training pool (N_PER_CLASS rows per mode); split 70/15/15 by the trainers
  dataset_natural.csv     independent set with the natural class mix (use it to report realistic performance)
  dataset_infeasible.csv  scenarios where NO mode is feasible (tests the validator's warning)
  scenario_summary.json   class mix, label-noise rate, rule statistics

How a scenario is built
  60% "real": a weather hour from a random buoy (waves, wind and rain stay jointly consistent) + a random AIS
              traffic sample. 40% "stress": sea state 0-6, rain up to 50 mm/h and traffic up to 200 vessels,
              to cover conditions the buoys and one-day AIS rarely contain.
  Mission inputs (always sampled): required range (log-uniform), required resolution (log-uniform), target RCS.
  Label: radar_common.label_scenarios() on the TRUE values; THEN sensor noise on the features and a few percent
  of label noise next to decision boundaries.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import radar_common as rc

# ---------------------------------------------------------------- sampling assumptions (all adjustable)
REAL_FRACTION = 0.60
RANGE_KM = (0.5, 100.0)          # clip for the required range. (Radar horizon limits ship targets to ~50 km.)
R_FRAC = (0.3, 1.3)              # required range = R_FRAC x (best clear-weather detection range for that target), log-uniform:
                                 # operators ask for ranges at which the target is detectable in good conditions, so most
                                 # requests sit near the capability boundary where weather decides the mode
RES_M = (12.0, 200.0)            # required separation, log-uniform; 12 m = finest the 80 ns mode can resolve
RCS_DBSM = (0.0, 35.0)           # small boat ... large ship, uniform
RAIN_WET_MMPH = 0.1              # below this the hour counts as dry
RAIN_PATH_FRAC = (0.2, 1.0)      # share of the required range that is inside rain
RAIN_PATH_MAX_KM = 30.0
STRESS_RAIN = [("dry", 0.30, 0.0, 0.0, "const"), ("light", 0.20, 0.1, 2.5, "uni"), ("moderate", 0.20, 2.5, 7.6, "uni"),
               ("heavy", 0.20, 7.6, 25.0, "log"), ("very_heavy", 0.10, 25.0, 50.0, "log")]
STRESS_N_VESSELS = (1, 200)      # vessels within 12 NM, log-uniform
STRESS_MINPAIR_M = (10.0, 500.0)
R12_M = 12 * 1852.0
LABEL_NOISE_TARGET = 0.04        # overall share of labels flipped to a neighbouring feasible mode
NEAR_DB = 1.0                    # "near a decision boundary" = within 1 dB


def loguni(rng, lo, hi, n):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n))


class Sampler:
    def __init__(self, hourly, traffic, real_fraction):
        self.h = hourly.reset_index(drop=True)
        self.stations = sorted(self.h["station"].unique())
        self.rows_of = {s: np.flatnonzero((self.h["station"] == s).to_numpy()) for s in self.stations}
        self.wind_of_state = {s: self.h.loc[self.h["sea_state"] == s, "wspd_ms"].to_numpy() for s in range(7)}
        t = traffic.copy()
        t["nearest_m"] = t["nearest_m"].fillna(rc.FILL_M)
        t["minpair_12nm_m"] = t["minpair_12nm_m"].fillna(rc.FILL_M)
        self.t = t[["n_12nm", "minpair_12nm_m", "nearest_m"]].to_numpy(float)
        self.real_fraction = real_fraction

    def _real(self, n, rng):
        st = rng.integers(0, len(self.stations), n)
        idx = np.empty(n, int)
        for k, s in enumerate(self.stations):
            m = st == k
            idx[m] = rng.choice(self.rows_of[s], size=int(m.sum()))
        h = self.h.iloc[idx]
        tr = self.t[rng.integers(0, len(self.t), n)]
        return pd.DataFrame({"source": "real", "station": h["station"].to_numpy(),
                             "wvht_m_true": h["wvht_m"].to_numpy(), "wspd_ms_true": h["wspd_ms"].to_numpy(),
                             "rain_mmph_true": h["rain_mmph"].to_numpy(), "sea_state_true": h["sea_state"].to_numpy(),
                             "n_12nm_true": tr[:, 0], "minpair_m_true": tr[:, 1], "nearest_m_true": tr[:, 2]})

    def _stress(self, n, rng):
        s = rng.integers(0, 7, n)                                            # sea state 0..6, uniform
        edges = np.array(rc.DOUGLAS_EDGES)
        lo = np.where(s >= 1, edges[np.clip(s - 1, 0, 8)], 0.0)
        hi = np.where(s >= 1, edges[np.clip(s, 0, 8)], 0.0)
        hs = rng.uniform(lo, hi)
        wind = np.empty(n)
        for k in range(7):
            m = s == k
            pool = self.wind_of_state.get(k, np.array([]))
            if k <= 1 or len(pool) == 0:
                wind[m] = rng.uniform(0, 3, m.sum()) if k <= 1 else 6 + 2 * k + rng.normal(0, 1.5, m.sum())
            else:
                wind[m] = rng.choice(pool, size=int(m.sum()))
        names = rng.choice(len(STRESS_RAIN), size=n, p=[c[1] for c in STRESS_RAIN])
        rain = np.zeros(n)
        for k, (_, _, a, b, kind) in enumerate(STRESS_RAIN):
            m = names == k
            if kind == "uni":
                rain[m] = rng.uniform(a, b, m.sum())
            elif kind == "log":
                rain[m] = loguni(rng, a, b, m.sum())
        nv = np.maximum(1, np.round(loguni(rng, *STRESS_N_VESSELS, n)))
        minpair = np.where(nv >= 2, loguni(rng, *STRESS_MINPAIR_M, n), rc.FILL_M)
        nearest = np.clip(R12_M * np.sqrt(-np.log(rng.uniform(1e-9, 1, n)) / nv), 20.0, rc.FILL_M)   # Poisson nearest neighbour
        return pd.DataFrame({"source": "stress", "station": "none", "wvht_m_true": hs, "wspd_ms_true": np.maximum(wind, 0),
                             "rain_mmph_true": rain, "sea_state_true": s, "n_12nm_true": nv,
                             "minpair_m_true": minpair, "nearest_m_true": nearest})

    def batch(self, n, rng):
        n_real = int(round(n * self.real_fraction))
        d = pd.concat([self._real(n_real, rng), self._stress(n - n_real, rng)], ignore_index=True)
        d = d.sample(frac=1.0, random_state=int(rng.integers(1 << 30))).reset_index(drop=True)
        n = len(d)
        d["rcs_dbsm_true"] = rng.uniform(*RCS_DBSM, n)
        r_det = rc.CAL_RANGE_KM * 10 ** ((d["rcs_dbsm_true"] - rc.CAL_RCS_DBSM + rc.best_clear_gain_db()) / 40)
        d["range_req_km"] = np.clip(r_det * loguni(rng, *R_FRAC, n), *RANGE_KM)
        d["res_req_m"] = loguni(rng, *RES_M, n)
        wet = d["rain_mmph_true"] > RAIN_WET_MMPH
        d["rain_path_km_true"] = np.where(
            wet, np.minimum(RAIN_PATH_MAX_KM, d["range_req_km"] * rng.uniform(*RAIN_PATH_FRAC, n)), 0.0)
        d["edge"] = (((d["rain_mmph_true"] >= 7.6) | (d["n_12nm_true"] >= 50)) & (d["range_req_km"] >= 20)).astype(int)
        lab = rc.label_scenarios(d)
        d["label_clean"] = lab["label"]
        d["margin_db"] = lab["margin"]
        d["rule_threshold"] = lab["rule_threshold"].astype(int)
        return d


def finalize(d, rng, noise_target):
    """Sensor noise on features, label noise near boundaries, engineered features (computed from OBSERVED values)."""
    d = rc.add_sensor_noise(d, rng)
    n = len(d)
    y = d["label_clean"].to_numpy(int)
    feas = rc.feasibility(d["range_req_km"], d["res_req_m"], d["minpair_m_true"], d["nearest_m_true"])
    near = (d["margin_db"].to_numpy() < NEAR_DB) & (y >= 0)
    p = min(0.5, noise_target / near.mean()) if near.any() else 0.0
    flip = near & (rng.random(n) < p)
    sign = rng.choice([-1, 1], n)
    c1, c2 = y + sign, y - sign
    ar = np.arange(n)
    ok1 = (c1 >= 0) & (c1 < rc.N_MODES) & feas[ar, np.clip(c1, 0, 4)]
    ok2 = (c2 >= 0) & (c2 < rc.N_MODES) & feas[ar, np.clip(c2, 0, 4)]
    new = np.where(ok1, c1, np.where(ok2, c2, y))
    d["label"] = np.where(flip, new, y)
    d["flipped"] = (d["label"] != d["label_clean"]).astype(int)
    return rc.add_engineered(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="processed")
    ap.add_argument("--out", default="scenarios")
    ap.add_argument("--per-class", type=int, default=8000)
    ap.add_argument("--natural", type=int, default=30000)
    ap.add_argument("--infeasible", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=200_000)
    ap.add_argument("--max-batches", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    hourly = pd.read_csv(Path(args.processed) / "hourly_sea_rain.csv")
    traffic = pd.read_csv(Path(args.processed) / "ais_traffic.csv")
    sampler = Sampler(hourly, traffic, REAL_FRACTION)
    ss = np.random.SeedSequence(args.seed)
    r_nat, r_pool, r_noise = [np.random.default_rng(s) for s in ss.spawn(3)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- natural-mix evaluation set (independent of the balanced pool) ----------------------------
    nat_all = sampler.batch(args.natural * 2, r_nat)
    nat_inf_share = float((nat_all["label_clean"] < 0).mean())
    nat = nat_all[nat_all["label_clean"] >= 0].head(args.natural).reset_index(drop=True)
    natural_mix = nat["label_clean"].value_counts(normalize=True).reindex(range(5), fill_value=0)

    # ---- class-balanced pool: keep generating until every mode has per_class rows -----------------
    keep = {k: [] for k in range(5)}
    count = np.zeros(5, int)
    inf_rows, n_inf = [], 0
    seen = 0
    pool_mix = np.zeros(5)
    for b in range(args.max_batches):
        d = sampler.batch(args.batch, r_pool)
        seen += len(d)
        for k in range(5):
            need = args.per_class - count[k]
            if need > 0:
                rows = d[d["label_clean"] == k].head(need)
                keep[k].append(rows)
                count[k] += len(rows)
        if n_inf < args.infeasible:
            rows = d[d["label_clean"] < 0].head(args.infeasible - n_inf)
            inf_rows.append(rows)
            n_inf += len(rows)
        pool_mix += np.bincount(d["label_clean"][d["label_clean"] >= 0], minlength=5)
        print(f"  batch {b + 1}: class counts {count.tolist()}  infeasible {n_inf}")
        if (count >= args.per_class).all() and n_inf >= args.infeasible:
            break
    short = [rc.MODE_NAMES[k] for k in range(5) if count[k] < args.per_class]
    if short:
        print("WARNING: could not reach the target count for", short, "- the dataset is imbalanced; "
              "relax the sampling ranges or raise --max-batches")
    pool = pd.concat([r for k in range(5) for r in keep[k]], ignore_index=True)
    pool = pool.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    inf = pd.concat(inf_rows, ignore_index=True) if inf_rows else pool.head(0)

    pool, nat, inf = finalize(pool, r_noise, LABEL_NOISE_TARGET), finalize(nat, r_noise, LABEL_NOISE_TARGET), \
        finalize(inf, r_noise, 0.0)
    pool.to_csv(out / "dataset.csv", index=False)
    nat.to_csv(out / "dataset_natural.csv", index=False)
    inf.to_csv(out / "dataset_infeasible.csv", index=False)

    # ---- how much does each input group actually decide the label? (answers the professor's question) ----
    def influence_of(df):
        y0 = df["label_clean"].to_numpy()
        def relabel(**over):
            d = df.copy()
            for k, v in over.items():
                d[k] = v
            return float((rc.label_scenarios(d)["label"] != y0).mean())
        return {
            "rain_removed": relabel(rain_mmph_true=0.0, rain_path_km_true=0.0),
            "sea_state_0": relabel(sea_state_true=0),
            "traffic_removed": relabel(minpair_m_true=rc.FILL_M, nearest_m_true=rc.FILL_M),
            "all_environment_removed": relabel(rain_mmph_true=0.0, rain_path_km_true=0.0, sea_state_true=0,
                                              minpair_m_true=rc.FILL_M, nearest_m_true=rc.FILL_M),
        }
    influence = {"natural_mix": influence_of(nat), "balanced_set": influence_of(pool)}

    summary = {
        "label_changed_share_when_input_removed": influence,
        "n_balanced": int(len(pool)), "n_natural": int(len(nat)), "n_infeasible": int(len(inf)),
        "scenarios_generated_for_pool": int(seen),
        "natural_class_mix": {rc.MODE_NAMES[k]: round(float(natural_mix[k]), 4) for k in range(5)},
        "natural_infeasible_share": nat_inf_share,
        "balanced_class_counts": {rc.MODE_NAMES[k]: int(count[k]) for k in range(5)},
        "label_noise_flipped_share": {"balanced": float(pool["flipped"].mean()), "natural": float(nat["flipped"].mean())},
        "share_labeled_by_threshold_rule": float(nat["rule_threshold"].mean()),
        "source_share_natural": nat["source"].value_counts(normalize=True).round(3).to_dict(),
        "edge_share_natural": float(nat["edge"].mean()),
        "sea_state_share_natural": nat["sea_state_true"].value_counts(normalize=True).sort_index().round(3).to_dict(),
    }
    (out / "scenario_summary.json").write_text(json.dumps({k: (v if not isinstance(v, dict) else {str(a): b for a, b in v.items()})
                                                          for k, v in summary.items()}, indent=2))
    print("\nNatural class mix (what the scenario distribution produces before balancing):")
    for k in range(5):
        print(f"  {rc.MODE_NAMES[k]:7s} {natural_mix[k]:6.1%}")
    print(f"No feasible mode in {nat_inf_share:.1%} of natural scenarios (validator warning case)")
    print("\nShare of labels that CHANGE when an input group is neutralised (how much the environment decides the mode):")
    print(f"  {'':26s}{'natural':>9s}{'balanced':>10s}")
    for k in influence["natural_mix"]:
        print(f"  {k:26s}{influence['natural_mix'][k]:9.1%}{influence['balanced_set'][k]:10.1%}")
    print()
    print(f"Labels flipped by boundary noise: balanced {summary['label_noise_flipped_share']['balanced']:.1%}, "
          f"natural {summary['label_noise_flipped_share']['natural']:.1%}")
    print(f"Labelled by the 'shortest sufficient' rule: {summary['share_labeled_by_threshold_rule']:.1%}; "
          f"by 'best SCNR' fallback: {1 - summary['share_labeled_by_threshold_rule']:.1%}")
    print(f"\nSaved {len(pool)} balanced, {len(nat)} natural, {len(inf)} infeasible rows to {out.resolve()}")


if __name__ == "__main__":
    main()
