#rfwave.py
import re
from pathlib import Path

import numpy as np

PLOTS_DIR = Path(__file__).resolve().parent / "plots"

FC_NOM     = 9.41e9          # spec: 9410 MHz
FC_TOL     = 30e6            # spec: +/- 30 MHz
SPC        = 10              # samples per carrier cycle (from the PDF)
FS_PER_SEC = 10**15          # VCD timescale is 1 fs

SPECS = {
    0b000: dict(name="Mode 1", pw=80,  prf=3000),
    0b001: dict(name="Mode 2", pw=800, prf=600),
    0b010: dict(name="Mode 3", pw=200, prf=2000),
    0b011: dict(name="Mode 4", pw=400, prf=1000),
    0b100: dict(name="Mode 5", pw=600, prf=800)
}


def carrier_pulse(fc, pw_ns, spc=SPC, amp=1.0, pad=1):
    """s(t) = TX(t) * cos(2*pi*fc*t), built over the gate window only."""
    if abs(fc - FC_NOM) > FC_TOL:
        raise ValueError(f"fc = {fc/1e6:.1f} MHz is outside 9410 +/- 30 MHz")
    fs = spc * fc                                   # 94.1 GS/s
    n = int(round(fs * pw_ns * 1e-9))               # samples inside the gate
    k = np.arange(-pad, n + pad)                    # sample index, gate starts at k = 0
    t = k / fs                                      # local time, starts at the gate rise
    tx = ((k >= 0) & (k < n)).astype(float)         # TX gate
    s = amp * tx * np.cos(2 * np.pi * fc * t)
    return t, tx, s, fs, n


def write_vcd(path, pulses, fc):
    """One VCD: txGate, rxBlank (wires) and rf (real) with a 1 fs timescale."""
    ev = []                                         # (time in fs, text)
    for p in pulses:
        pw_ns = p["g_fall"] - p["g_rise"]
        t, tx, s, fs, n = carrier_pulse(fc, pw_ns)
        r_fs = p["g_rise"] * 10**6                  # 1 ns = 10^6 fs
        ev.append((r_fs, "1a"))
        ev.append((p["g_fall"] * 10**6, "0a"))
        ev.append((p["b_rise"] * 10**6, "1b"))
        ev.append((p["b_fall"] * 10**6, "0b"))
        for ti, si in zip(t, s):
            ev.append((r_fs + int(round(ti * FS_PER_SEC)), f"r{float(si) + 0.0:.6g} c"))
    ev.sort(key=lambda e: e[0])

    with open(path, "w") as f:
        f.write("$timescale 1 fs $end\n$scope module radar $end\n"
                "$var wire 1 a txGate $end\n$var wire 1 b rxBlank $end\n"
                "$var real 1 c rf $end\n$upscope $end\n$enddefinitions $end\n"
                "#0\n$dumpvars\n0a\n0b\nr0 c\n$end\n")
        last = None
        for t_fs, txt in ev:
            if t_fs != last:
                f.write(f"#{t_fs}\n")
                last = t_fs
            f.write(txt + "\n")


def verify(pulses, fc, guard_ns):
    for code, spec in SPECS.items():
        ps = [p for p in pulses if p["mode"] == code]
        if not ps:
            continue
        rises = np.array([p["g_rise"] for p in ps])
        pw    = np.array([p["g_fall"] - p["g_rise"] for p in ps])
        lead  = np.array([p["g_rise"] - p["b_rise"] for p in ps])
        guard = np.array([p["b_fall"] - p["g_fall"] for p in ps])
        pri   = np.diff(rises)

        t, tx, s, fs, n = carrier_pulse(fc, pw.mean())
        seg = s[1:1 + n]                                        # samples inside the gate only
        cycles = int(np.sum((seg[:-1] < 0) & (seg[1:] >= 0)))   # one rising zero-crossing per cycle
        expected = fc * pw.mean() * 1e-9

        rows = [("Pulses captured", len(ps) >= 3, f"{len(ps)}"),
                ("PW equals spec", bool(np.all(pw == spec["pw"])), f"{pw.mean():.0f} ns")]
        if len(pri):
            prf = 1e9 / pri.mean()
            rows.append(("PRF within 0.1% of nominal",
                         abs(prf - spec["prf"]) / spec["prf"] < 1e-3, f"{prf:.3f} Hz"))
        rows += [("RX blank starts with/before gate", bool(np.all(lead >= 0)), f"{lead.min()} ns"),
                 ("RX guard >= configured", bool(np.all(guard >= guard_ns)), f"{guard.min()} ns"),
                 ("Carrier in 9410 +/- 30 MHz", abs(fc - FC_NOM) <= FC_TOL, f"{fc/1e6:.1f} MHz"),
                 ("Cycles = fc x PW (+/-1)", abs(cycles - expected) <= 1,
                  f"{cycles} vs {expected:.1f}"),
                 ("Samples = fs x PW", n == round(fs * pw.mean() * 1e-9), f"{n}")]

        print(f"\n{spec['name']}")
        print(f"{'Check':36s}{'Result':8s}Detail")
        for label, ok, detail in rows:
            print(f"{label:36s}{'PASS' if ok else 'FAIL':8s}{detail}")


def plot(pulses, fc):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, len(SPECS), figsize=(4 * len(SPECS), 6))
    for col, (code, spec) in enumerate(SPECS.items()):
        p = next((q for q in pulses if q["mode"] == code), None)
        if p is None:
            continue
        pw_ns = p["g_fall"] - p["g_rise"]
        pad = int(round(10e-9 * SPC * fc))                      # 10 ns of idle each side
        t, tx, s, fs, n = carrier_pulse(fc, pw_ns, pad=pad)
        tn = t * 1e9
        ax[0, col].plot(tn, s, lw=0.3, label="s(t)")
        ax[0, col].plot(tn, tx, "r", lw=1.2, label="TX gate")
        ax[0, col].set(title=f"{spec['name']}: full pulse ({pw_ns} ns)", xlabel="ns")
        ax[0, col].legend(loc="upper right")
        a, b = pad - 10, pad + 45                               # idle + about 4.5 cycles
        ax[1, col].plot(tn[a:b], s[a:b], "o-", ms=3, label="samples")
        ax[1, col].step(tn[a:b], tx[a:b], "r", where="post", label="TX gate")
        ax[1, col].set(title="Zoom: gate edge and first cycles", xlabel="ns")
        ax[1, col].legend(loc="upper right")
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / "waveforms.png", dpi=150)
    plt.show()


def rf_events(pulses, fc):
    """NumPy carrier samples as (time in fs, VCD text) events."""
    ev = []
    for p in pulses:
        pw_ns = p["g_fall"] - p["g_rise"]
        t, tx, s, fs, n = carrier_pulse(fc, pw_ns)
        r_fs = p["g_rise"] * 10**6
        for ti, si in zip(t, s):
            ev.append((r_fs + int(round(ti * FS_PER_SEC)), f"r{float(si) + 0.0:.6g} ZZ"))
    return ev



def add_rf_to_vcd(vcd_path, pulses, fc, echo_ev=None):
    text = open(vcd_path).read()
    text = re.sub(r"\$timescale.*?\$end", "$timescale 1 fs $end", text, flags=re.S)
    head, body = text.split("$enddefinitions $end", 1)
    head += "$scope module rf_carrier $end\n$var real 1 ZZ rf $end\n$upscope $end\n"
    if echo_ev:
        import radar_sim
        head += radar_sim.VCD_HEADER

    groups, t = {0: []}, 0
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.fullmatch(r"#\d+", line):
            t = int(line[1:]) * 10**6            # ns -> fs
            groups.setdefault(t, [])
        else:
            groups[t].append(line)

    groups[0].append("r0 ZZ")
    if echo_ev:
        groups[0] += radar_sim.VCD_INIT
    for t_fs, txt in rf_events(pulses, fc) + (echo_ev or []):
        groups.setdefault(t_fs, []).append(txt)

    with open(vcd_path, "w") as f:
        f.write(head + "$enddefinitions $end\n")
        for t_fs in sorted(groups):
            f.write(f"#{t_fs}\n")
            for l in groups[t_fs]:
                f.write(l + "\n")
