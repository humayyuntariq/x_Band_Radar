"""Spectral analysis of an ideal rectangular 9.41 GHz radar pulse (Mode 1 and Mode 2).

One program, both modes. Pulse widths come from SPECS by default, or from the
measured TX-gate edges of the MyHDL timing simulation (see pw_from_pulses).
Scope: single isolated rectangular pulse. No shaping, chirp, echo or Doppler.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FC       = 9.41e9          # carrier (Hz)
FC_TOL   = 30e6            # spec: 9410 MHz +/- 30 MHz
SPC      = 10              # samples per carrier cycle -> fs = 10 * fc
PLOTS_ROOT = Path(__file__).resolve().parent / "plots"   # same root folder rfwave.py uses
OUT      = PLOTS_ROOT                                    # set per run in run()
NFFT_EXP = 24              # NFFT = 2**24 -> fine frequency grid (~5.6 kHz bins)

SPECS = {                  # name -> pulse width (s), plot half-span (MHz)
    "Mode 1": dict(pw=80e-9,  span=50, tag="Mode 1"),
    "Mode 2": dict(pw=800e-9, span=8,  tag="Mode 2"),
    "Mode 3": dict(pw=200e-9, span=20, tag="Mode 3"),
    "Mode 4": dict(pw=400e-9, span=12, tag="Mode 4"),
    "Mode 5": dict(pw=600e-9, span=10, tag="Mode 5")
}


def pw_from_pulses(pulses, mode_code):
    """PW (s) of one mode from the MyHDL gate edges: pulses = [{mode, g_rise, g_fall} in ns]."""
    w = [p["g_fall"] - p["g_rise"] for p in pulses if p["mode"] == mode_code]
    return float(np.mean(w)) * 1e-9


def rf_pulse(pw, fc=FC):
    """Rectangular RF pulse x(t) = cos(2*pi*fc*t), 0 <= t < PW, sampled at 10*fc."""
    if abs(fc - 9.41e9) > FC_TOL:
        raise ValueError("fc outside 9410 +/- 30 MHz")
    fs = SPC * fc
    n = int(round(pw * fs))
    t = np.arange(n) / fs
    return t, np.cos(2 * np.pi * fc * t), fs


def spectrum(x, fs, fc=FC):
    nfft = 2 ** NFFT_EXP
    S = np.fft.rfft(x, n=nfft)                       # zero-padded real FFT
    f = np.fft.rfftfreq(nfft, d=1 / fs)
    mag = np.abs(S)
    db = 20 * np.log10(np.maximum(mag / mag.max(), 1e-12))
    off = (f - fc) / 1e6                             # MHz from carrier
    return off, db, fs / nfft


def first_nulls(off, db, window=60.0):
    """Nearest FFT minimum on each side of the carrier (first rise in dB walking outward)."""
    i0 = int(np.argmax(db))
    res = []
    for step in (-1, +1):
        i = i0
        while abs(off[i + step] - off[i0]) < window and db[i + step] <= db[i]:
            i += step
        res.append(off[i])
    return i0, res[0], res[1]


def first_sidelobe_db(off, db, i_lo, i_hi):
    """Highest level outside the main lobe, up to 3 null-widths from the carrier."""
    w = (off[i_hi] - off[i_lo])
    m = (off > off[i_hi]) & (off < off[i_hi] + w)
    return db[m].max()


def analyze(name, pw, fc=FC, pw_spec=None):
    t, x, fs = rf_pulse(pw, fc)
    off, db, binhz = spectrum(x, fs, fc)
    i0, lo, hi = first_nulls(off, db)
    theory = 1 / pw / 1e6
    return dict(name=name, pw=pw, pw_spec=pw_spec or pw, t=t, x=x, fs=fs, off=off, db=db, bin_khz=binhz / 1e3,
                peak_off=off[i0], null_lo=lo, null_hi=hi, bw_sim=hi - lo,
                bw_theory=2 * theory, null_theory=theory,
                err_pct=abs((hi - lo) - 2 * theory) / (2 * theory) * 100,
                sidelobe=first_sidelobe_db(off, db, np.searchsorted(off, lo), np.searchsorted(off, hi)),
                n=len(x))


def plot_time(r, fc=FC, tag=""):
    """Separate time-domain figure per mode: full pulse (with 10 ns idle each side) + zoomed carrier."""
    fs, n, pad = r["fs"], r["n"], int(round(10e-9 * SPC * fc))
    k = np.arange(-pad, n + pad)
    tn = k / fs * 1e9
    gate = ((k >= 0) & (k < n)).astype(float)
    s = gate * np.cos(2 * np.pi * fc * k / fs)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.5))
    ax[0].plot(tn, s, lw=0.3, label="s(t)")
    ax[0].plot(tn, gate, "r", lw=1.2, label="TX gate")
    ax[0].set(title=f"{r['name']}: rectangular RF pulse, PW = {r['pw']*1e9:.0f} ns "
                    f"({n} samples at {fs/1e9:.1f} GS/s)", xlabel="time (ns)", ylabel="amplitude")
    fig.suptitle(r.get("src", ""), fontsize=9, y=0.995)
    ax[0].legend(loc="upper right")
    a, b = pad - 10, pad + 45
    ax[1].plot(tn[a:b], s[a:b], "o-", ms=3, label="samples (10 per cycle)")
    ax[1].step(tn[a:b], gate[a:b], "r", where="post", label="TX gate")
    ax[1].set(title="Zoom: gate edge and first carrier cycles", xlabel="time (ns)", ylabel="amplitude")
    ax[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / f"{r['name'].replace(' ', '').lower()}_time.png", dpi=150)
    plt.close(fig)


def plot_spectrum(r, span):
    m = np.abs(r["off"]) <= span
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(r["off"][m], r["db"][m], lw=0.9, label="FFT magnitude")
    ax.axvline(-r["null_theory"], ls="--", color="gray")
    ax.axvline(+r["null_theory"], ls="--", color="gray",
               label=f"Theoretical first nulls (±{r['null_theory']:.3f} MHz)")
    ax.plot([r["null_lo"], r["null_hi"]], [np.interp(r["null_lo"], r["off"][m], r["db"][m])] * 2,
            "rv", ms=6, label=f"Measured nulls ({r['null_lo']:.3f}, {r['null_hi']:+.3f} MHz)")
    ax.set(xlabel="Offset from 9.41 GHz (MHz)", ylabel="Normalized magnitude (dB)",
           title=f"Spectrum of rectangular radar pulse: {r['name']} (PW = {r['pw']*1e9:.0f} ns)",
           ylim=(-80, 3))
    fig.suptitle(r.get("src", ""), fontsize=9, y=0.995)
    ax.grid(True)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / f"{r['name'].replace(' ', '').lower()}_spectrum.png", dpi=150)
    plt.close(fig)


def plot_comparison(results, span=50):
    fig, ax = plt.subplots(figsize=(10, 5))
    for r, c in zip(results, ("tab:blue", "tab:orange", "", "tab:green", "tab:red", "tab:purple")):
        m = np.abs(r["off"]) <= span
        ax.plot(r["off"][m], r["db"][m], lw=0.9, color=c or None, label=f"{r['name']} (PW {r['pw']*1e9:.0f} ns)")
    ax.set(xlabel="Offset from 9.41 GHz (MHz)", ylabel="Normalized magnitude (dB)",
           title="Normalized spectra: the shorter pulse has the wider main lobe", ylim=(-80, 3))
    fig.suptitle(results[0].get("src", ""), fontsize=9, y=0.995)
    ax.grid(True)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "comparison_spectrum.png", dpi=150)
    plt.close(fig)


def table_rows(r):
    """(label, unit, decimals, theory, simulated) for one mode.
    THEORY = formulas f_c +/- 1/PW and B = 2/PW.  SIMULATED = measured from the FFT of the sampled pulse."""
    th = r["null_theory"]                                    # MHz
    return [
        ("Pulse width",          "ns",  1, r["pw_spec"] * 1e9, r["pw"] * 1e9),
        ("Carrier centre",       "GHz", 6, FC / 1e9,           FC / 1e9 + r["peak_off"] / 1e3),
        ("Lower first null",     "GHz", 6, FC / 1e9 - th / 1e3, FC / 1e9 + r["null_lo"] / 1e3),
        ("Upper first null",     "GHz", 6, FC / 1e9 + th / 1e3, FC / 1e9 + r["null_hi"] / 1e3),
        ("Lower null offset",    "MHz", 4, -th,                r["null_lo"]),
        ("Upper null offset",    "MHz", 4, +th,                r["null_hi"]),
        ("Main-lobe width (B)",  "MHz", 4, r["bw_theory"],     r["bw_sim"]),
        ("Peak sidelobe",        "dB",  2, -13.26,             r["sidelobe"]),
    ]


def report(results):
    """One verification table: Theory | Simulated | Difference for each mode.
    Written to plots/verification_table.csv (all modes side by side) and .txt (readable copy)."""
    rows = {r["name"]: table_rows(r) for r in results}
    labels = [(l, u, d) for l, u, d, _, _ in rows[results[0]["name"]]]

    head = ["Quantity", "Unit"]
    for r in results:
        head += [f"{r['name']} Theory", f"{r['name']} Simulated", f"{r['name']} Difference"]
    csv = [",".join(head)]
    for i, (lab, u, d) in enumerate(labels):
        cells = [lab, u]
        for r in results:
            _, _, _, a, b = rows[r["name"]][i]
            diff = (b - a) * 1e6 if u == "GHz" else (b - a)          # GHz differences shown in kHz
            dd = 1 if u == "GHz" else d
            cells += [f"{a:.{d}f}", f"{b:.{d}f}", f"{diff:+.{dd}f}" + (" kHz" if u == "GHz" else "")]
        csv.append(",".join(cells))
    cells = ["Percentage error of B", "%"]
    for r in results:
        cells += ["0", f"{r['err_pct']:.3f}", f"{r['err_pct']:.3f}"]
    csv.append(",".join(cells))
    cells = ["FFT bin spacing", "kHz"]
    for r in results:
        cells += [f"{r['bin_khz']:.2f}", f"{r['bin_khz']:.2f}", ""]
    csv.append(",".join(cells))
    ratio = results[0]["bw_sim"] / results[1]["bw_sim"]
    csv.append(f"Mode 1 / Mode 2 width ratio,,10.000,{ratio:.3f},{ratio - 10:+.3f},,,")

    csv.append("Pulse width source,," + results[0].get("src", ""))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verification_table.csv").write_text("\n".join(csv) + "\n")

    txt = [f"Source of pulse width: {results[0].get('src', '')}"]
    for r in results:
        txt.append(f"\n{r['name']}   PW = {r['pw']*1e9:.0f} ns   N = {r['n']} samples   "
                   f"FFT bin spacing = {r['bin_khz']:.2f} kHz")
        txt.append(f"{'Quantity':22s}{'Theory':>14s}{'Simulated':>14s}{'Difference':>16s}")
        for lab, u, d, a, b in rows[r["name"]]:
            diff = f"{(b - a) * 1e6:+.1f} kHz" if u == "GHz" else f"{b - a:+.{d}f} {u}"
            txt.append(f"{lab + ' (' + u + ')':22s}{a:>14.{d}f}{b:>14.{d}f}{diff:>16s}")
        txt.append(f"{'Percentage error (%)':22s}{'0':>14s}{r['err_pct']:>14.3f}")
    txt.append(f"\nMode 1 / Mode 2 main-lobe width ratio: theory 10.000, simulated {ratio:.3f}")
    text = "\n".join(txt)
    print(text)
    (OUT / "verification_table.txt").write_text(text + "\n")


def run(pws=None, fc=FC, out_dir=None):
    """pws=None      -> stand-alone run: PW taken from SPECS      -> plots/spectral_standalone/
    pws={name: PW s} -> testbench run: PW measured from the MyHDL timing simulation -> plots/spectral_from_testbench/"""
    global OUT
    standalone = pws is None
    OUT = Path(out_dir) if out_dir else (
        PLOTS_ROOT / ("spectral" if standalone else "testbench"))
    OUT.mkdir(parents=True, exist_ok=True)
    src = "PW from SPECS (stand-alone)" if standalone else "PW measured from MyHDL timing simulation"
    results = []
    for name, spec in SPECS.items():
        pw = (pws or {}).get(name, spec["pw"])
        r = analyze(name, pw, fc, pw_spec=spec["pw"])
        r["src"] = src
        plot_time(r, fc)
        plot_spectrum(r, spec["span"])
        results.append(r)
    plot_comparison(results)
    report(results)
    print(f"\nSource of pulse width: {src}\nOutputs saved in: {OUT}")
    return results


if __name__ == "__main__":
    run()