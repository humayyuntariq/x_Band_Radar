"""download_data.py - fetch diverse buoy + rain data for the radar mode recommender.

For every station in STATIONS it downloads
  * NDBC historical standard-met files (one per year) and joins them into  data/ndbc/<station>.txt
  * Open-Meteo (ERA5) hourly rain at the station's coordinates       into  data/rain/<station>.csv
Stations whose NDBC file is missing for a year are skipped with a message, so listing extra
candidates is safe.

Usage:
  python3 download_data.py                         # all stations, years 2022 2023
  python3 download_data.py --years 2022 2023 2024
  python3 download_data.py --stations 46042 44013 42001

Uses the system `curl` (same tool you already used for Open-Meteo), so no extra Python packages.
Coordinates below are approximate (ERA5 grid is ~25 km): check them on each station's NDBC page if you like.
"""
import argparse
import gzip
import subprocess
import time
from pathlib import Path

NDBC_URL = "https://www.ndbc.noaa.gov/data/historical/stdmet/{st}h{year}.txt.gz"
RAIN_URL = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
            "&start_date={start}&end_date={end}&hourly=rain&timezone=GMT&format=csv")

# station id -> (climate regime, lat, lon). Chosen to cover different sea / rain / wind regimes.
STATIONS = {
    "46042": ("Monterey CA - mild Pacific coast, winter rain", 36.785, -122.398),
    "44013": ("Boston MA - NW Atlantic, cold, winter storms", 42.35, -70.65),
    "42001": ("Mid Gulf of Mexico - subtropical, convective rain, hurricanes", 25.89, -89.66),
    "41008": ("Gray's Reef GA - sheltered shallow coast, calm", 31.40, -80.87),
    "46001": ("Gulf of Alaska - high latitude, rough, persistent rain", 56.30, -148.02),
    "51101": ("NW Hawaii - tropical mid-Pacific, trade-wind showers, swell", 24.32, -162.08),
    "51002": ("SW Hawaii - tropical mid-Pacific", 17.09, -157.81),
}


def curl(url, out_path, retries=3):
    """Download url -> out_path. Returns True on success, False on HTTP error (e.g. 404)."""
    for attempt in range(retries):
        r = subprocess.run(["curl", "-sS", "-f", "-L", "-m", "120", "-o", str(out_path), url],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return True
        if r.returncode == 22:                       # HTTP 4xx/5xx
            if "404" in r.stderr:
                return False
            time.sleep(5 * (attempt + 1))            # 429/5xx: wait and retry
        else:
            time.sleep(3 * (attempt + 1))
    return False


def read_text(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", errors="replace") as f:
        return f.read().splitlines()


def combine_years(files, out_path):
    """Join yearly NDBC files. Keep the 2 header lines once; drop repeated headers from later years."""
    header, body = [], []
    for fp in files:
        lines = [l for l in read_text(fp) if l.strip()]
        if not header:
            header = [l for l in lines if l.startswith("#")][:2]
        body += [l for l in lines if not l.startswith("#")]
    out_path.write_text("\n".join(header + body) + "\n")
    return len(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2022, 2023])
    ap.add_argument("--stations", nargs="+", default=list(STATIONS))
    ap.add_argument("--out", default="data")
    ap.add_argument("--ndbc-url", default=NDBC_URL)
    ap.add_argument("--rain-url", default=RAIN_URL)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "ndbc").mkdir(parents=True, exist_ok=True)
    (out / "rain").mkdir(parents=True, exist_ok=True)
    tmp = out / "_tmp"
    tmp.mkdir(exist_ok=True)
    start, end = f"{min(args.years)}-01-01", f"{max(args.years)}-12-31"
    ok = []

    for st in args.stations:
        if st not in STATIONS:
            print(f"[{st}] not in STATIONS table: add its name and coordinates first")
            continue
        label, lat, lon = STATIONS[st]
        print(f"[{st}] {label}")
        files = []
        for y in args.years:
            fp = tmp / f"{st}_{y}.txt.gz"
            if curl(args.ndbc_url.format(st=st, year=y), fp):
                files.append(fp)
            else:
                print(f"    no NDBC file for {y} (skipped)")
        if len(files) < len(args.years):
            print(f"    -> station dropped: needs every requested year ({len(args.years)})")
            continue
        n = combine_years(files, out / "ndbc" / f"{st}.txt")
        rain_fp = out / "rain" / f"{st}.csv"
        if not curl(args.rain_url.format(lat=lat, lon=lon, start=start, end=end), rain_fp):
            print("    -> rain download failed (free API limit? wait a minute and rerun)")
            continue
        print(f"    NDBC rows: {n}   rain saved: {rain_fp}")
        ok.append(st)
        time.sleep(2)                                # be polite to the free API

    for f in tmp.glob("*"):
        f.unlink()
    tmp.rmdir()
    print(f"\nReady stations ({len(ok)}): {' '.join(ok)}")
    if ok:
        nd = " ".join(f"{out}/ndbc/{s}.txt" for s in ok)
        rn = " ".join(f"{out}/rain/{s}.csv" for s in ok)
        print(f"\nNext:\npython3 preprocess.py --ndbc {nd} \\\n    --rain {rn} \\\n    --ais ais-2023-12-30.csv.zst")


if __name__ == "__main__":
    main()
