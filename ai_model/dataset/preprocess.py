"""preprocess.py - Stage 3 data preparation for the radar mode recommender.

Reads real-data files and turns them into calibration tables for scenario_gen.py:
  1. NDBC standard met files (one per buoy)      -> wave height, wind speed -> Douglas sea state
  2. Open-Meteo rain CSVs (one per buoy)         -> rain rate (mm/h)
  3. MarineCadastre AIS day file(s)              -> traffic density and target spacing

Several buoys can be given so the data covers different climates. --ndbc and --rain are paired by order.

Usage:
  python3 preprocess.py --ndbc data/ndbc/46042.txt data/ndbc/44013.txt --rain data/rain/46042.csv data/rain/44013.csv \\
                        --ais ais-2023-12-30.csv.zst
  (download_data.py prints the exact command for the stations it downloaded.)
  Optional: --bbox LAT_MIN LAT_MAX LON_MIN LON_MAX   restrict AIS to a region (default: whole file)

Outputs go to ./processed/:
  hourly_sea_rain.csv     hourly rows: station, time, wvht_m, wspd_ms, rain_mmph, douglas_raw, sea_state
  ais_traffic.csv         own-ship-centred samples: vessel counts, density, nearest and min pair spacing
  calibration.json        overall + per-station statistics, ITU-R P.838-3 coefficients at 9.41 GHz
  preprocess_summary.png  plots for the report

Needs: numpy, pandas, scipy, matplotlib, and zstandard (only for .zst files: pip install zstandard).
"""
import argparse
import gzip
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

FC_GHZ = 9.41
NM_M = 1852.0
EARTH_R = 6371000.0
SEED = 42

# Douglas sea scale (WMO sea-state code) upper limits of significant wave height, metres.
DOUGLAS_EDGES = [0.0, 0.1, 0.5, 1.25, 2.5, 4.0, 6.0, 9.0, 14.0]


# ----------------------------------------------------------------------------------------
# ITU-R P.838-3 specific rain attenuation  gamma = k * R^alpha  (dB/km)
# ----------------------------------------------------------------------------------------
_KH = dict(a=[-5.33980, -0.35351, -0.23789, -0.94158], b=[-0.10008, 1.26970, 0.86036, 0.64552],
           c=[1.13098, 0.45400, 0.15354, 0.16817], m=-0.18961, c0=0.71147)
_KV = dict(a=[-3.80595, -3.44965, -0.39902, 0.50167], b=[0.56934, -0.22911, 0.73042, 1.07319],
           c=[0.81061, 0.51059, 0.11899, 0.27195], m=-0.16398, c0=0.63297)
_AH = dict(a=[-0.14318, 0.29591, 0.32177, -5.37610, 16.1721], b=[1.82442, 0.77564, 0.63773, -0.96230, -3.29980],
           c=[-0.55187, 0.19822, 0.13164, 1.47828, 3.43990], m=0.67849, c0=-1.95537)
_AV = dict(a=[-0.07771, 0.56727, -0.20238, -48.2991, 48.5833], b=[2.33840, 0.95545, 1.14520, 0.791669, 0.791459],
           c=[-0.76284, 0.54039, 0.26809, 0.116226, 0.116479], m=-0.053739, c0=0.83433)


def _gauss_sum(p, f_ghz):
    x = np.log10(f_ghz)
    return sum(a * np.exp(-((x - b) / c) ** 2) for a, b, c in zip(p["a"], p["b"], p["c"])) + p["m"] * x + p["c0"]


def itu_p838_3(f_ghz=FC_GHZ):
    """k, alpha for H, V and circular polarisation on a horizontal path (ship radar)."""
    kH, kV = 10 ** _gauss_sum(_KH, f_ghz), 10 ** _gauss_sum(_KV, f_ghz)
    aH, aV = _gauss_sum(_AH, f_ghz), _gauss_sum(_AV, f_ghz)
    k_c = (kH + kV) / 2
    a_c = (kH * aH + kV * aV) / (2 * k_c)
    return dict(kH=kH, aH=aH, kV=kV, aV=aV, k_circ=k_c, alpha_circ=a_c)


# ----------------------------------------------------------------------------------------
# 1. NDBC standard meteorological file
# ----------------------------------------------------------------------------------------
def load_ndbc(path):
    """Robust to .gz, MM fill values and header lines repeated inside joined multi-year files."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", errors="replace") as f:
        lines = [l for l in f.read().splitlines() if l.strip()]
    header = next(l for l in lines if l.startswith("#"))            # first '#' line holds the column names
    data = [l for l in lines if not l.startswith("#")]
    df = pd.read_csv(io.StringIO("\n".join([header] + data)), sep=r"\s+", header=0)
    df.columns = [c.lstrip("#") for c in df.columns]
    year = pd.to_numeric(df["YY"], errors="coerce")
    year = np.where(year < 100, year + 2000, year)
    minute = df["mm"] if "mm" in df.columns else 0
    df["time"] = pd.to_datetime(dict(year=year, month=df["MM"], day=df["DD"], hour=df["hh"], minute=minute),
                                errors="coerce")
    out = df[["time", "WVHT", "WSPD"]].dropna(subset=["time"]).copy()
    for col in ("WVHT", "WSPD"):                                      # NDBC fills missing values with 99.0 / 999 / MM
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out.loc[out[col] >= 98, col] = np.nan
    out = out.rename(columns={"WVHT": "wvht_m", "WSPD": "wspd_ms"})
    out["hour"] = out["time"].dt.floor("h")
    return out.groupby("hour", as_index=False)[["wvht_m", "wspd_ms"]].mean()


# ----------------------------------------------------------------------------------------
# 2. Open-Meteo rain CSV (has a few metadata lines before the real header)
# ----------------------------------------------------------------------------------------
def load_rain(path):
    lines = Path(path).read_text().splitlines()
    start = next((i for i, l in enumerate(lines) if l.split(",")[0].strip().lower() in ("time", "date", "datetime")), 0)
    df = pd.read_csv(path, skiprows=start)
    tcol = df.columns[0]
    cols = [c for c in df.columns if "precip" in c.lower()] or [c for c in df.columns if "rain" in c.lower()]
    if not cols:
        raise SystemExit(f"No precipitation/rain column in {path}. Columns found: {list(df.columns)}")
    out = pd.DataFrame({"hour": pd.to_datetime(df[tcol]).dt.floor("h"),
                        "rain_mmph": pd.to_numeric(df[cols[0]], errors="coerce")})   # hourly total (mm) = mean rate (mm/h)
    return out.groupby("hour", as_index=False)["rain_mmph"].mean()


def douglas(hs):
    return np.digitize(hs, DOUGLAS_EDGES, right=True)


# ----------------------------------------------------------------------------------------
# 3. AIS traffic: own-ship-centred snapshots
# ----------------------------------------------------------------------------------------
def load_ais(path, bbox, min_sog):
    head = pd.read_csv(path, nrows=0).columns
    low = {c.lower(): c for c in head}
    # accepts both the old (BaseDateTime, LAT, LON) and new (base_date_time, latitude, longitude) schemas
    aliases = {"mmsi": ("mmsi",), "basedatetime": ("basedatetime", "base_date_time"),
               "lat": ("lat", "latitude"), "lon": ("lon", "longitude"), "sog": ("sog",)}
    need = {}
    for key, names in aliases.items():
        hit = next((low[n] for n in names if n in low), None)
        if hit is not None:
            need[key] = hit
    if len(need) < 5:
        raise SystemExit(f"AIS file is missing columns. Found: {list(head)}")
    parts = []
    for ch in pd.read_csv(path, usecols=list(need.values()), chunksize=1_000_000):
        ch.columns = [next(k for k, v in need.items() if v == c) for c in ch.columns]
        ch = ch[(ch.lat.abs() <= 90) & (ch.lon.abs() <= 180) & (ch.sog >= min_sog) & (ch.sog < 102)]
        if bbox:
            ch = ch[(ch.lat >= bbox[0]) & (ch.lat <= bbox[1]) & (ch.lon >= bbox[2]) & (ch.lon <= bbox[3])]
        parts.append(ch)
    df = pd.concat(parts, ignore_index=True)
    df["t"] = pd.to_datetime(df["basedatetime"])
    df["bin"] = df["t"].dt.floor("10min")
    df = df.sort_values("t").drop_duplicates(["bin", "mmsi"], keep="last")      # last report per vessel per 10 min
    return df


def xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return EARTH_R * np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def chord_to_arc(c):
    return 2 * EARTH_R * np.arcsin(np.clip(c / (2 * EARTH_R), 0, 1))


def ais_samples(df, n_own, radii_nm, rng, stratify=True):
    """Own-ship-centred samples. With stratify=True every 1-degree cell gets equal weight, so busy ports
    do not dominate and quiet coastal/offshore areas are represented too."""
    r_big = max(radii_nm) * NM_M
    rows = []
    for b, g in df[df["bin"].dt.minute == 0].groupby("bin"):        # one snapshot per hour
        if len(g) < 2:
            continue
        lat, lon = g["lat"].to_numpy(), g["lon"].to_numpy()
        pts = xyz(lat, lon)
        tree = cKDTree(pts)
        cell = pd.Series(list(zip(np.floor(lat), np.floor(lon))))
        w = (1.0 / cell.map(cell.value_counts()).to_numpy()) if stratify else np.ones(len(g))
        own = rng.choice(len(g), size=min(n_own, len(g)), replace=False, p=w / w.sum())
        for i in own:
            idx = [j for j in tree.query_ball_point(pts[i], r=r_big) if j != i]
            d = chord_to_arc(np.linalg.norm(pts[idx] - pts[i], axis=1)) if idx else np.array([])
            row = {"snapshot": b, "own_lat": float(lat[i]), "own_lon": float(lon[i]),
                   "nearest_m": d.min() if len(d) else np.nan}
            for r in radii_nm:
                m = d <= r * NM_M
                row[f"n_{r}nm"] = int(m.sum())
                row[f"density_{r}nm_per_km2"] = m.sum() / (np.pi * (r * NM_M / 1000) ** 2)
            r0 = radii_nm[0]
            sel = np.array(idx)[d <= r0 * NM_M] if len(d) else np.array([], dtype=int)
            row[f"minpair_{r0}nm_m"] = (chord_to_arc(pdist(pts[sel[:1500]]).min())
                                        if len(sel) >= 2 else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
def q(s, levels=(0.05, 0.25, 0.5, 0.75, 0.95, 0.99)):
    s = pd.Series(s).dropna()
    return {f"p{int(l * 100)}": float(s.quantile(l)) for l in levels} if len(s) else {}


def rain_fit(rain_series):
    wet = rain_series[rain_series > 0.1]
    return ({"mu": float(np.log(wet).mean()), "sigma": float(np.log(wet).std())} if len(wet) > 5 else {})


def station_summary(h):
    wet = h.loc[h.rain_mmph > 0.1, "rain_mmph"]
    return {"n_hours": int(len(h)), "days": int((h.time.max() - h.time.min()).days),
            "hs_median_m": float(h.wvht_m.median()), "hs_p95_m": float(h.wvht_m.quantile(0.95)),
            "wind_p95_ms": float(h.wspd_ms.quantile(0.95)),
            "p_wet": float((h.rain_mmph > 0.1).mean()), "rain_p99_mmph": float(h.rain_mmph.quantile(0.99)),
            "rain_max_mmph": float(h.rain_mmph.max()), "wet_lognormal": rain_fit(h.rain_mmph),
            "sea_state_freq": {int(k): round(float(v), 4) for k, v in
                               h.sea_state.value_counts(normalize=True).sort_index().items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndbc", nargs="+", required=True, help="one NDBC file per buoy")
    ap.add_argument("--rain", nargs="+", required=True, help="one Open-Meteo CSV per buoy, same order as --ndbc")
    ap.add_argument("--ais", nargs="+", required=True, help="one or more MarineCadastre daily files")
    ap.add_argument("--out", default="processed")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"))
    ap.add_argument("--min-sog", type=float, default=0.5, help="drop vessels slower than this (knots)")
    ap.add_argument("--own-per-snapshot", type=int, default=50)
    ap.add_argument("--no-stratify", action="store_true", help="sample own ships uniformly instead of per 1-degree cell")
    ap.add_argument("--max-sea-state", type=int, default=6)
    args = ap.parse_args()
    if len(args.ndbc) != len(args.rain):
        raise SystemExit("--ndbc and --rain must list the same number of files (paired by order).")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    warnings = []

    # ---- sea + rain per station -------------------------------------------------------
    frames = []
    for nf, rf in zip(args.ndbc, args.rain):
        st = Path(nf).stem
        sea, rain = load_ndbc(nf), load_rain(rf)
        valid = sea.wvht_m.notna().mean()
        m = sea.merge(rain, on="hour", how="inner").rename(columns={"hour": "time"})
        m = m.dropna(subset=["wvht_m", "wspd_ms", "rain_mmph"])
        print(f"[{st}] buoy {sea.hour.min():%Y-%m-%d}..{sea.hour.max():%Y-%m-%d}  WVHT valid {valid:.0%}   "
              f"rain {rain.hour.min():%Y-%m-%d}..{rain.hour.max():%Y-%m-%d}   merged hours: {len(m)}")
        if valid < 0.5:
            warnings.append(f"{st}: more than half of WVHT is missing - this station does not report waves; dropped.")
            continue
        if m.empty:
            warnings.append(f"{st}: buoy and rain files do not overlap in time; dropped.")
            continue
        m.insert(0, "station", st)
        frames.append(m)
    if not frames:
        raise SystemExit("\nERROR: no usable buoy/rain pairs. See the messages above.")
    hourly = pd.concat(frames, ignore_index=True)
    hourly["douglas_raw"] = douglas(hourly["wvht_m"])
    hourly["sea_state"] = hourly["douglas_raw"].clip(upper=args.max_sea_state)
    hourly.to_csv(out / "hourly_sea_rain.csv", index=False)
    per_station = {st: station_summary(g) for st, g in hourly.groupby("station")}
    print(f"\nMerged hourly rows, all stations: {len(hourly)}")
    for st, s in per_station.items():
        if s["days"] < 300:
            warnings.append(f"{st}: only {s['days']} days covered - seasonal bias.")
    if len(per_station) < 3:
        warnings.append("Fewer than 3 stations: the weather data represents very few climates.")

    # ---- AIS --------------------------------------------------------------------------
    traffic_parts = []
    for af in args.ais:
        ais = load_ais(af, args.bbox, args.min_sog)
        print(f"AIS {Path(af).name}: {len(ais)} vessel-reports (10-min dedup), {ais.mmsi.nunique()} vessels")
        traffic_parts.append(ais_samples(ais, args.own_per_snapshot, [12, 24], rng, stratify=not args.no_stratify))
    traffic = pd.concat(traffic_parts, ignore_index=True)
    traffic.to_csv(out / "ais_traffic.csv", index=False)
    print(f"AIS own-ship samples: {len(traffic)}")
    if args.bbox is None:
        warnings.append("AIS used without --bbox: samples cover all of the file's waters (diverse, but US-only).")
    warnings.append("AIS dates differ from the buoy/rain period: AIS is used only for the traffic "
                    "distribution, not paired in time.")

    # ---- calibration summary ----------------------------------------------------------
    itu = itu_p838_3(FC_GHZ)
    wet = hourly.loc[hourly.rain_mmph > 0.1, "rain_mmph"]
    cal = {
        "stations": list(per_station),
        "n_hours_merged": int(len(hourly)),
        "wave_height_m": q(hourly.wvht_m),
        "wind_ms": q(hourly.wspd_ms),
        "sea_state_freq": {int(k): round(float(v), 4) for k, v in
                           hourly.sea_state.value_counts(normalize=True).sort_index().items()},
        "rain": {"p_wet_gt_0p1_mmph": float((hourly.rain_mmph > 0.1).mean()),
                 "wet_hour_quantiles_mmph": q(wet), "wet_hour_lognormal": rain_fit(hourly.rain_mmph),
                 "max_mmph": float(hourly.rain_mmph.max())},
        "per_station": per_station,
        "itu_p838_3_at_9p41GHz": {k: round(float(v), 5) for k, v in itu.items()},
        "ais": {"n_samples": int(len(traffic)),
                "n_in_12nm": q(traffic["n_12nm"]), "density_12nm_per_km2": q(traffic["density_12nm_per_km2"]),
                "nearest_m": q(traffic["nearest_m"]), "minpair_12nm_m": q(traffic["minpair_12nm_m"]),
                "frac_zero_vessels_12nm": float((traffic["n_12nm"] == 0).mean())},
        "warnings": warnings,
    }
    (out / "calibration.json").write_text(json.dumps(cal, indent=2))

    # ---- console table ----------------------------------------------------------------
    print(f"\n{'station':8s}{'hours':>7s}{'Hs med':>8s}{'Hs p95':>8s}{'wind p95':>9s}{'P(wet)':>8s}{'rain p99':>9s}{'rain max':>9s}")
    for st, s in per_station.items():
        print(f"{st:8s}{s['n_hours']:7d}{s['hs_median_m']:8.2f}{s['hs_p95_m']:8.2f}{s['wind_p95_ms']:9.1f}"
              f"{s['p_wet']:8.1%}{s['rain_p99_mmph']:9.2f}{s['rain_max_mmph']:9.1f}")

    # ---- plot -------------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 3, figsize=(14, 7.5))
        sts = list(per_station)
        ax[0, 0].boxplot([hourly.loc[hourly.station == s, "wvht_m"] for s in sts], showfliers=False)
        ax[0, 0].set_xticks(range(1, len(sts) + 1), sts)
        ax[0, 0].set(title="Wave height by station", ylabel="Hs (m)")
        ax[0, 1].bar(cal["sea_state_freq"].keys(), cal["sea_state_freq"].values())
        ax[0, 1].set(title="Douglas sea state (all stations)", xlabel="state")
        ax[0, 2].boxplot([hourly.loc[hourly.station == s, "wspd_ms"] for s in sts], showfliers=False)
        ax[0, 2].set_xticks(range(1, len(sts) + 1), sts)
        ax[0, 2].set(title="Wind speed by station", ylabel="m/s")
        ax[1, 0].hist(wet, 40); ax[1, 0].set(title=f"Rain rate, wet hours (P(wet)={cal['rain']['p_wet_gt_0p1_mmph']:.0%})",
                                             xlabel="mm/h", yscale="log")
        ax[1, 1].hist(traffic["n_12nm"], 40); ax[1, 1].set(title="Vessels within 12 NM (AIS)", xlabel="count", yscale="log")
        ax[1, 2].hist(np.log10(traffic["minpair_12nm_m"].dropna().clip(lower=1)), 40)
        ax[1, 2].set(title="Min spacing between vessels, 12 NM", xlabel="log10 metres")
        for a in (ax[0, 0], ax[0, 2]):
            a.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(out / "preprocess_summary.png", dpi=150)
    except Exception as e:                                     # plot is optional
        print("Plot skipped:", e)

    print("\nITU-R P.838-3 at 9.41 GHz:", {k: round(float(v), 5) for k, v in itu.items()})
    for w in warnings:
        print("WARNING:", w)
    print(f"\nSaved to {out.resolve()}")


if __name__ == "__main__":
    main()
