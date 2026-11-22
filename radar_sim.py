"""radar_sim.py - physics-based echo simulation driven by the TWG timing from the MyHDL run.

Chain: scene (random R, fixed RCS, v) -> monostatic radar range equation
       -> delayed, phase-rotated echo (complex baseband)
       -> receiver (active while rxEnable is HIGH, i.e. after rxBlank falls) + thermal & environment noise
       -> matched filter -> Doppler FFT over the pulses on target -> CA-CFAR -> R and v estimates.

Exports two VCD signals in module echo_sim:
  - receiver (real): environment + thermal noise the entire time rxEnable is HIGH, with a radar-equation
                     spike at the target echo location (0 while rxEnable is LOW / rxBlank is HIGH).
  - echoTrue (wire): binary (0/1) detection signal showing where the target object is detected.
"""
import numpy as np

C, NM, K_B, T0 = 299_792_458.0, 1852.0, 1.380649e-23, 290.0
FS = 100e6                     # baseband sample rate = TWG clock (10 ns, 1.5 m per sample)
TS = 1 / FS
PFA = 1e-8                     # CFAR design false-alarm probability per cell
BEAM_DEG, RPM = 1.2, 24.0      # antenna beamwidth and scan rate
DWELL = BEAM_DEG / (6 * RPM)   # time the beam stays on the target: 8.33 ms -> pulses on target

FIXED_RCS = 50.0               # fixed target Radar Cross Section area (m^2) across all modes
RADAR = dict(pt=10e3, g_db=30.0, loss_db=3.0, nf_db=5.0, env_db=3.0)

# Each mode draws a random target location (in NM) within its valid range window; RCS is fixed at FIXED_RCS.
TARGETS = {
    0b000: dict(r_range_nm=(2.0, 3.2), rcs=FIXED_RCS, v=4.0),   # Mode 1 (PW = 80 ns)
    0b010: dict(r_range_nm=(2.6, 4.2), rcs=FIXED_RCS, v=4.0),   # Mode 3 (PW = 200 ns)
    0b011: dict(r_range_nm=(3.2, 5.0), rcs=FIXED_RCS, v=4.0),   # Mode 4 (PW = 400 ns)
    0b100: dict(r_range_nm=(3.6, 5.6), rcs=FIXED_RCS, v=4.0),   # Mode 5 (PW = 600 ns)
    0b001: dict(r_range_nm=(4.0, 6.2), rcs=FIXED_RCS, v=4.0),   # Mode 2 (PW = 800 ns)
}
NAMES = {0b000: "Mode 1", 0b001: "Mode 2", 0b010: "Mode 3", 0b011: "Mode 4", 0b100: "Mode 5"}

VCD_HEADER = ("$scope module echo_sim $end\n"
              "$var real 1 ZR receiver $end\n"
              "$var wire 1 ZT echoTrue $end\n$upscope $end\n")
VCD_INIT = ["r0 ZR", "0ZT"]


def timing(pulses, code):
    """Pulse timing of one mode as measured on the HDL gate/blank edges (ns -> s)."""
    all_ps = sorted(pulses, key=lambda p: p["g_rise"])
    idx_map = {id(p): i for i, p in enumerate(all_ps)}
    ps = [p for p in all_ps if p["mode"] == code]
    rise = np.array([p["g_rise"] for p in ps], float)
    b_rise = np.array([p["b_rise"] for p in ps], float)
    b_fall = np.array([p["b_fall"] for p in ps], float)
    if len(rise) > 1:
        pri = float(np.mean(np.diff(rise))) * 1e-9
    elif ps and "pri_ns" in ps[0]:
        pri = float(ps[0]["pri_ns"]) * 1e-9
    else:
        pri = {0b000: 333.33e-6, 0b001: 1666.66e-6, 0b010: 500e-6, 0b011: 1000e-6, 0b100: 1250e-6}.get(code, 333.33e-6)
    off = float(np.mean(b_fall - rise)) * 1e-9                  # receiver opens (rxEnable goes HIGH)

    # For each pulse, rxEnable stays HIGH from b_fall until the next pulse's b_rise (or end of PRI for last pulse)
    next_b_rise = []
    for p in ps:
        i = idx_map[id(p)]
        if i + 1 < len(all_ps):
            next_b_rise.append(float(all_ps[i + 1]["b_rise"]))
        else:
            next_b_rise.append(float(p["g_rise"]) + pri * 1e9)

    return dict(rise=rise, b_rise=b_rise, b_fall=b_fall, next_b_rise=np.array(next_b_rise, float),
                pw=float(np.mean([p["g_fall"] - p["g_rise"] for p in ps])) * 1e-9,
                pri=pri, off=off)


def budget(code, fc, pw, rng=None, r_nm=None, tim=None):
    """Monostatic radar range equation + thermal & environment noise budget."""
    t = TARGETS[code]
    if r_nm is None:
        r_lo, r_hi = t["r_range_nm"]
        if tim is not None:
            r_min_valid = (C * (tim["off"] + 2 * pw) / 2) / NM
            r_max_valid = (C * max(tim["pri"] - 2 * pw, tim["off"] + 4 * pw) / 2) / NM
            if r_hi > 0.85 * r_max_valid or r_lo < r_min_valid:
                r_lo = max(r_min_valid, min(r_lo, 0.25 * r_max_valid))
                r_hi = max(r_lo * 1.15, min(r_hi, 0.80 * r_max_valid))
        r_nm = float(rng.uniform(r_lo, r_hi)) if rng is not None else 0.5 * (r_lo + r_hi)
    lam = C / fc
    R = r_nm * NM
    rcs = float(t["rcs"])
    g, L, F, E = (10 ** (RADAR[k] / 10) for k in ("g_db", "loss_db", "nf_db", "env_db"))

    # Monostatic radar range equation: Pr = (Pt * G^2 * lambda^2 * RCS) / ((4*pi)^3 * R^4 * L)
    pr = RADAR["pt"] * (g ** 2) * (lam ** 2) * rcs / (((4 * np.pi) ** 3) * (R ** 4) * L)

    n0_thermal = K_B * T0 * F                                   # receiver thermal noise spectral density
    n0_env = n0_thermal * (E - 1.0)                             # environment / clutter noise spectral density
    n0 = n0_thermal + n0_env                                    # total noise spectral density
    return dict(R=R, r_nm=r_nm, rcs=rcs, v=t["v"], lam=lam, pr=pr,
                n0_thermal=n0_thermal, n0_env=n0_env, n0=n0, snr=pr * pw / n0)


def simulate(code, fc, tim, rng, npulse, r_nm=None):
    """Received I/Q for npulse pulses. Fast time starts when rxEnable goes HIGH (rxBlank falls)."""
    b = budget(code, fc, tim["pw"], rng=rng, r_nm=r_nm, tim=tim)
    N = int(round((tim["pri"] - tim["off"]) / TS))
    lo = tim["off"] + np.arange(N) * TS                         # sample n covers [lo, lo + TS)
    sig = np.zeros((npulse, N), complex)
    for m in range(npulse):
        tau = 2 * (b["R"] + b["v"] * m * tim["pri"]) / C        # target range delay for pulse m
        ov = np.clip(np.minimum(lo + TS, tau + tim["pw"]) - np.maximum(lo, tau), 0, TS) / TS
        sig[m] = np.sqrt(b["pr"]) * np.exp(-2j * np.pi * fc * tau) * ov

    # Independent thermal noise + environment noise across the entire rxEnable HIGH window
    s_th = np.sqrt(b["n0_thermal"] * FS / 2)
    s_env = np.sqrt(b["n0_env"] * FS / 2)
    noise_th = s_th * (rng.standard_normal(sig.shape) + 1j * rng.standard_normal(sig.shape))
    noise_env = s_env * (rng.standard_normal(sig.shape) + 1j * rng.standard_normal(sig.shape))
    noise = noise_th + noise_env
    rx = sig + noise
    b["sig"] = sig
    b["noise"] = noise
    return rx, b


def process(rx, tim, b, pfa=PFA):
    M = rx.shape[0]
    npw = int(round(tim["pw"] / TS))
    cs = np.concatenate([np.zeros((M, 1), complex), np.cumsum(rx, axis=1)], axis=1)
    y = (cs[:, npw:] - cs[:, :-npw]) / np.sqrt(npw)             # matched filter (unit-energy rect)
    nfft = max(16, 1 << (M - 1).bit_length())
    p = np.abs(np.fft.fft(y, n=nfft, axis=0)) ** 2 / M          # range-Doppler power
    L = p.shape[1]
    T, G = 16 * npw, npw + 2                                    # CA-CFAR training / guard cells per side
    c = np.concatenate([np.zeros((nfft, 1)), np.cumsum(p, axis=1)], axis=1)
    i = np.arange(L)
    l0, l1 = np.clip(i - G - T, 0, L), np.clip(i - G, 0, L)
    r0, r1 = np.clip(i + G + 1, 0, L), np.clip(i + G + T + 1, 0, L)
    cnt = np.maximum((l1 - l0) + (r1 - r0), 1)
    noise = np.maximum(((c[:, l1] - c[:, l0]) + (c[:, r1] - c[:, r0])) / cnt, 1e-30)
    ne = np.maximum(cnt / npw, 2)                               # independent cells (MF noise is correlated)
    thr = ne * (pfa ** (-1 / ne) - 1) * noise
    k, n = np.unravel_index(np.argmax(p), p.shape)
    a = np.sqrt(p[k])
    nh = n + (a[n + 1] - a[n - 1]) / (2 * (a[n] - min(a[n - 1], a[n + 1]))) if 0 < n < L - 1 else float(n)
    tau_hat = tim["off"] + nh * TS                              # triangle interpolation of the MF peak
    f = np.fft.fftfreq(nfft, tim["pri"])[k]
    return dict(det=bool(p[k, n] > thr[k, n]), tau_hat=tau_hat, R_hat=C * tau_hat / 2,
                v_hat=-f * b["lam"] / 2, snr_db=10 * np.log10(p[k, n] / noise[k, n]),
                n=int(n), k=int(k), p=p, thr=thr, y=y)


def vcd_events(tim, rx, b, r, rng=None, nshow=None):
    """Generates (time_fs, vcd_cmd) for echo_sim.receiver and echo_sim.echoTrue.
    - While rxEnable is LOW (rxBlank is HIGH, [b_rise, b_fall)): receiver = 0.
    - While rxEnable is HIGH ([b_fall, next_b_rise)): receiver shows continuous environment + thermal
      noise at every clock sample (10 ns), with a radar-equation spike at the target echo location.
    - echoTrue is a binary signal (0/1) pulsing HIGH where the target object is detected.
    """
    ev = []
    npw = int(round(tim["pw"] / TS))
    sc_noise = 1.0 / np.sqrt(b["n0"] * FS / 2)                  # normalizes noise to ~1.0 RMS units
    sig_if = b["sig"] * np.sqrt(npw)                            # radar-equation echo in matched IF bandwidth
    rx_env = np.abs((sig_if + b["noise"]) * sc_noise)           # receiver envelope: noise + radar-eq target spike

    if nshow is None:
        nshow = len(tim["rise"])
    n_pulses = min(nshow, len(tim["rise"]), rx_env.shape[0])
    for m in range(n_pulses):
        g_rise_ns = tim["rise"][m]
        b_rise_ns = tim["b_rise"][m]
        b_fall_ns = tim["b_fall"][m]
        end_ns = tim["next_b_rise"][m]

        # When rxBlank rises (rxEnable goes LOW), receiver outputs 0
        ev.append((int(round(b_rise_ns * 1e6)), "r0 ZR"))

        # Binary detection signal echoTrue pulses HIGH where the object is detected
        if r["det"]:
            nc = int(round((r["tau_hat"] - tim["off"]) / TS))
            t_det_start = int(round((b_fall_ns + nc * (TS * 1e9)) * 1e6))
            t_det_end = int(round((b_fall_ns + (nc + npw) * (TS * 1e9)) * 1e6))
            ev.append((t_det_start, "1ZT"))
            ev.append((t_det_end, "0ZT"))

        # Emit receiver signal (environment + thermal noise, plus target spike) for entire rxEnable HIGH duration
        n_listen = int(round((end_ns - b_fall_ns) / (TS * 1e9)))
        row = rx_env[m]
        if n_listen > len(row) and rng is not None:
            extra = np.abs(rng.standard_normal(n_listen - len(row)) + 1j * rng.standard_normal(n_listen - len(row)))
            row = np.concatenate([row, extra])
        else:
            row = row[:n_listen]

        base_fs = int(round(b_fall_ns * 1e6))
        step_fs = int(round(TS * 1e15))                         # 10 ns = 10,000,000 fs
        for n, val in enumerate(row.tolist()):
            ev.append((base_fs + n * step_fs, f"r{val:.4g} ZR"))

        # Ensure receiver drops to 0 right as the next pulse's rxBlank rises (rxEnable goes LOW)
        ev.append((int(round(end_ns * 1e6)), "r0 ZR"))

    return ev


def plot(name, tim, b, r, out="plots/echo"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    Path(out).mkdir(parents=True, exist_ok=True)
    db = lambda a: 10 * np.log10(a / (b["n0"] * FS) + 1e-12)
    R = C * (tim["off"] + np.arange(r["p"].shape[1]) * TS) / 2
    f = np.fft.fftshift(np.fft.fftfreq(r["p"].shape[0], tim["pri"]))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(R / NM, db(r["p"][r["k"]]), lw=.5, label="range profile (target Doppler bin)")
    ax[0].plot(R / NM, db(r["thr"][r["k"]]), "r", lw=.8, label=f"CFAR threshold (Pfa {PFA:g})")
    ax[0].axvline(b["R"] / NM, ls="--", c="gray", label=f"true range ({b['r_nm']:.2f} NM)")
    ax[0].set(title=f"{name}: range profile (RCS = {b['rcs']:.0f} m²)", xlabel="range (NM)", ylabel="dB over noise")
    ax[0].legend()
    ax[1].plot(f, np.fft.fftshift(db(r["p"][:, r["n"]])), "o-", ms=3)
    ax[1].axvline(-2 * b["v"] / b["lam"], ls="--", c="gray", label="true Doppler")
    ax[1].set(title=f"{name}: Doppler at the target range", xlabel="Hz", ylabel="dB over noise")
    ax[1].legend()
    fig.tight_layout(); fig.savefig(Path(out) / f"{name.replace(' ', '').lower()}_echo.png", dpi=130)
    plt.close(fig)


def run_all(pulses, fc, seed=None, plots=True):
    rng = np.random.default_rng(seed)
    results, events = {}, []

    # Show receiver noise during the initial startup window [0, first b_rise) while rxEnable is HIGH
    if pulses:
        first_b_rise_ns = int(round(min(p["b_rise"] for p in pulses)))
        for t_ns in range(0, first_b_rise_ns, int(round(TS * 1e9))):
            val = float(np.abs(rng.standard_normal() + 1j * rng.standard_normal()))
            events.append((t_ns * 10**6, f"r{val:.4g} ZR"))

    print(f"{'Mode':7s}{'M':>3s}{'RCS m2':>8s}{'R true':>9s}{'tau us':>8s}{'Pr dBm':>8s}{'SNR/p':>7s}{'SNR/CPI':>8s}"
          f"{'Det':>5s}{'R est':>9s}{'err m':>7s}{'v est':>7s}{'v true':>7s}{'blind m':>8s}")
    for code in sorted(NAMES, key=NAMES.get):
        if not any(p["mode"] == code for p in pulses):
            continue
        tim = timing(pulses, code)
        M = max(2, len(tim["rise"]), int(DWELL / tim["pri"] + 1e-6))  # pulses while the beam is on target
        rx, b = simulate(code, fc, tim, rng, M)
        r = process(rx, tim, b)
        results[code] = (tim, b, r)
        print(f"{NAMES[code]:7s}{M:3d}{b['rcs']:8.1f}{b['R']:9.0f}{2 * b['R'] / C * 1e6:8.2f}{10 * np.log10(b['pr'] * 1e3):8.1f}"
              f"{10 * np.log10(b['snr']):7.1f}{10 * np.log10(b['snr'] * M):8.1f}{'YES' if r['det'] else 'NO':>5s}"
              f"{r['R_hat']:9.1f}{r['R_hat'] - b['R']:7.2f}{r['v_hat']:7.2f}{b['v']:7.2f}{C * tim['off'] / 2:8.1f}")
        events += vcd_events(tim, rx, b, r, rng=rng)
        if plots:
            plot(NAMES[code], tim, b, r)
    return results, events


def monte_carlo(pulses, fc, code, trials=30, seed=None):
    """Pd and RMS range error over independent random target locations and noise realisations."""
    tim, rng, hit, err = timing(pulses, code), np.random.default_rng(seed), 0, []
    M = max(2, int(DWELL / tim["pri"] + 1e-6))
    for _ in range(trials):
        rx, b = simulate(code, fc, tim, rng, M)
        r = process(rx, tim, b)
        if r["det"]:
            hit += 1
            err.append(r["R_hat"] - b["R"])
    return hit / trials, (float(np.sqrt(np.mean(np.square(err)))) if err else float("nan"))
