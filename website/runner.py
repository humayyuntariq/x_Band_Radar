"""Parametrised testbench for the X-band radar timing engine.

Builds the same blocks twg.py builds (prfEngine, pwCrtl, rxBlanking) but feeds them the
PW and PRI you ask for instead of the two fixed modes in rangeSelector.py, runs the
MyHDL simulation, adds the 9.41 GHz carrier with rfwave.add_rf_to_vcd, and checks the result.

Run it by hand:
    python runner.py --pw-ns 80 --pri-us 333.33 --out runs/test/radar.vcd
app.py runs this same script as a subprocess for the web page.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")   # plots are saved to files, no window

# The timing blocks (prfEngine, pwCrtl, rxBlanking, rfwave) live in the project folder one level up.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from myhdl import (block, always, instance, delay, now, Signal, intbv,
                   ResetSignal, StopSimulation)

from prfEngine import prfEngine
from pwCrtl import pwCrtl
from rxBlanking import rxBlanking
import rfwave
import surfer_view
try:
    import radar_sim                       # echo simulation (project root)
except Exception as e:                     # keep the timing run working without it
    radar_sim, _RADAR_SIM_ERR = None, e
from radar_params import resolve, ParamError, CLK_NS


# PW (ns) -> 3-bit mode code of the five fixed modes (same table as rangeSelector)
MODE_BY_PW = {80: 0b000, 800: 0b001, 200: 0b010, 400: 0b011, 600: 0b100}


def mode_for(pw_ns):
    return MODE_BY_PW[min(MODE_BY_PW, key=lambda k: abs(k - pw_ns))]


@block
def testbench(p, gates, blanks):
    clk = Signal(bool(0))
    rst = ResetSignal(0, active=0, isasync=False)
    txGate = Signal(bool(0))
    rxBlank = Signal(bool(0))
    rxEnable = Signal(bool(0))
    txTrigger = Signal(bool(0))
    mode = Signal(intbv(mode_for(p["pw_ns"]))[3:])      # shown in the waveform, like twg.py's mode port
    prfCountMax = Signal(intbv(p["prf_count_max"], min=0, max=200001))
    pwCountMax = Signal(intbv(p["pw_count_max"], min=0, max=201))

    # Same wiring as twg.py, with constant count-max values instead of rangeSelector
    uprf = prfEngine(clk, rst, prfCountMax, txTrigger)
    upw = pwCrtl(clk, rst, txTrigger, pwCountMax, txGate)
    urxb = rxBlanking(clk, rst, txTrigger, txGate, rxBlank, rxEnable, p["guard_cycles"])

    @always(delay(CLK_NS // 2))
    def clkgen():
        clk.next = not clk

    @instance
    def stim():
        mode.next = mode_for(p["pw_ns"])
        yield clk.negedge
        rst.next = 0
        yield delay(2 * CLK_NS)
        rst.next = 1
        for i in range(p["pulses"]):
            yield txGate.posedge
            print(f"  pulse {i + 1} starts at {now()} ns")
        yield txGate.negedge
        rem_cycles = max(p["guard_cycles"] + 5, p["pri_cycles"] - p["pw_cycles"] - 5)
        yield delay(rem_cycles * CLK_NS)
        raise StopSimulation()

    @instance
    def gate_mon():
        while True:
            yield txGate.posedge
            r = now()
            yield txGate.negedge
            gates.append((r, now()))

    @instance
    def blank_mon():
        while True:
            yield rxBlank.posedge
            r = now()
            yield rxBlank.negedge
            blanks.append((r, now()))

    @always(clk.negedge)
    def chk():
        assert rxEnable != rxBlank, f"rxEnable and rxBlank equal at {now()} ns"

    return uprf, upw, urxb, clkgen, stim, gate_mon, blank_mon, chk


def verify(p, pulses, fc):
    """Checks against the parameters you asked for. Returns a list of (label, ok, detail)."""
    rises = np.array([q["g_rise"] for q in pulses])
    pw = np.array([q["g_fall"] - q["g_rise"] for q in pulses])
    lead = np.array([q["g_rise"] - q["b_rise"] for q in pulses])
    guard = np.array([q["b_fall"] - q["g_fall"] for q in pulses])
    pri = np.diff(rises)

    t, tx, s, fs, n = rfwave.carrier_pulse(fc, float(pw.mean()))
    seg = s[1:1 + n]
    cycles = int(np.sum((seg[:-1] < 0) & (seg[1:] >= 0)))
    expected = fc * pw.mean() * 1e-9

    rows = [("Pulses captured", len(pulses) == p["pulses"], f"{len(pulses)} of {p['pulses']}"),
            ("PW matches setting", bool(np.all(pw == p["pw_ns"])),
             f"{pw.mean():.0f} ns vs {p['pw_ns']:.0f} ns")]
    if len(pri):
        rows.append(("PRI matches setting", bool(np.all(pri == p["pri_us"] * 1000)),
                     f"{pri.mean() / 1000:.3f} us vs {p['pri_us']:.3f} us"))
    else:
        rows.append(("PRI matches setting", True,
                     f"{p['pri_us']:.3f} us (1 pulse)"))
    rows += [("RX blank starts with/before gate", bool(np.all(lead >= 0)), f"{lead.min()} ns"),
             ("RX guard >= configured", bool(np.all(guard >= p["guard_cycles"] * CLK_NS)),
              f"{guard.min()} ns"),
             ("Cycles = fc x PW (+/-1)", abs(cycles - expected) <= 1,
              f"{cycles} vs {expected:.1f}"),
             ("Samples = fs x PW", n == round(fs * pw.mean() * 1e-9), f"{n}")]
    return rows


def run_echo(pulses, fc, pw_ns, out_dir):
    """Echo simulation from the measured pulses. Returns (echo_events, echo_info), or (None, None) if it could not run.
    radar_sim.run_all(pulses, fc) -> (results, echo_events); it saves plots/echo/*.png in the
    current folder, so the call runs inside the run folder to keep the project folder clean."""
    if radar_sim is None:
        print(f"  WARNING: echo simulation skipped, radar_sim not importable: {_RADAR_SIM_ERR}")
        return None, None
    mode = mode_for(pw_ns)
    for p_ in pulses:
        p_.setdefault("mode", mode)          # radar_sim / rfwave pick their spec by mode
    here = os.getcwd()
    try:
        os.chdir(out_dir)
        res, echo_ev = radar_sim.run_all(pulses, fc)
        echo_info = None
        if mode in res:
            tim, b, r = res[mode]
            echo_info = {
                "det": bool(r["det"]),
                "r_true_m": round(float(b["R"]), 1),
                "r_true_nm": round(float(b.get("r_nm", b["R"] / 1852.0)), 2),
                "r_est_m": round(float(r["R_hat"]), 1),
                "err_m": round(float(r["R_hat"] - b["R"]), 2),
                "rcs_m2": round(float(b.get("rcs", 50.0)), 1),
                "snr_pulse_db": round(float(10 * np.log10(b["snr"])), 1),
                "snr_cpi_db": round(float(r["snr_db"]), 1),
                "tau_us": round(float(2 * b["R"] / 299_792_458.0 * 1e6), 2),
                "blind_m": round(float(299_792_458.0 * tim["off"] / 2), 1),
            }
        return echo_ev, echo_info
    except Exception as e:
        print(f"  WARNING: echo simulation failed ({type(e).__name__}: {e}); carrier only")
        return None, None
    finally:
        os.chdir(here)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pw-ns", type=float, required=True)
    ap.add_argument("--pri-us", type=float, required=True)
    ap.add_argument("--pulses", type=int, default=3)
    ap.add_argument("--guard", type=int, default=5, help="RX guard in clock cycles")
    ap.add_argument("--fc-mhz", type=float, default=9410.0)
    ap.add_argument("--no-echo", action="store_true", help="skip the echo simulation")
    ap.add_argument("--out", required=True, help="output VCD path, e.g. runs/test/radar.vcd")
    a = ap.parse_args()

    try:
        p = resolve(a.pw_ns, a.pri_us, a.pulses, a.guard, a.fc_mhz)
    except ParamError as e:
        print(f"ERROR: {e}")
        return 2

    print("Parameters applied to the timing engine")
    print(f"  PW   {p['pw_ns']:.0f} ns  ({p['pw_cycles']} cycles, pwCountMax = {p['pw_count_max']})")
    print(f"  PRI  {p['pri_us']:.3f} us  ({p['pri_cycles']} cycles, prfCountMax = {p['prf_count_max']})")
    print(f"  PRF  {p['prf_hz']:.3f} Hz   dR {p['delta_r_m']:.2f} m   R {p['range_km']:.2f} km")
    if abs(p["pw_ns"] - p["requested_pw_ns"]) > 1e-6 or abs(p["pri_us"] - p["requested_pri_us"]) > 1e-6:
        print(f"  note: values snapped to the {CLK_NS} ns clock grid "
              f"(requested PW {p['requested_pw_ns']:g} ns, PRI {p['requested_pri_us']:g} us)")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fc = p["fc_hz"]

    print("Running MyHDL simulation")
    gates, blanks = [], []
    tb = testbench(p, gates, blanks)
    tb.config_sim(trace=True, filename=str(out.with_suffix("")), tracebackup=False, timescale="1ns")
    duration = ((p["pulses"] + 2) * p["pri_cycles"] + 200) * CLK_NS
    try:
        tb.run_sim(duration=duration, quiet=1)
    except AssertionError as e:
        print(f"FAIL simulation self-check: {e}")
        tb.quit_sim()
        return 1
    tb.quit_sim()

    count = min(len(gates), len(blanks))
    pulses = [dict(g_rise=g[0], g_fall=g[1], b_rise=b[0], b_fall=b[1],
                   pri_ns=p["pri_cycles"] * CLK_NS)
              for g, b in zip(gates[:count], blanks[:count])]
    if not pulses:
        print("ERROR: no pulses were captured")
        return 1

    echo_ev, echo_info = None, None
    if not a.no_echo:
        print("Running echo simulation (radar_sim)")
        echo_ev, echo_info = run_echo(pulses, fc, p["pw_ns"], out.parent)

    print("Adding 9.41 GHz carrier" + (" and echo signals" if echo_ev is not None else "") + " to the VCD")
    if echo_ev is not None:
        rfwave.add_rf_to_vcd(str(out), pulses, fc, echo_ev)
    else:
        rfwave.add_rf_to_vcd(str(out), pulses, fc)

    with open(out) as _f:
        _head = []
        for _l in _f:
            _head.append(_l)
            if "$enddefinitions" in _l:
                break
    _has_echo = "echo_sim" in "".join(_head)
    print(f"  echo_sim signals in the VCD: {'yes' if _has_echo else 'NO'}"
          f"  (echo events returned: {None if echo_ev is None else len(echo_ev)})")

    print("Writing Surfer view (zoom, analog rf, signal heights)")
    try:
        view = surfer_view.write_view(out)
        print(f"  signals: {', '.join(view['signals'])}")
        if view["missing"]:
            print(f"  not in this VCD, skipped: {', '.join(view['missing'])}")
        if view["window"]:
            t0, t1 = view["window"]
            print(f"  zoom: TX trigger rise to TX gate fall ({(t1 - t0) / 1e6:.0f} ns)")
        else:
            print("  WARNING: no complete pulse found, Surfer will zoom to fit")
    except Exception as e:
        print(f"  WARNING: Surfer view not written: {e}")

    print("Verification")
    rows = verify(p, pulses, fc)
    for label, ok, detail in rows:
        print(f"  {'PASS' if ok else 'FAIL'}  {label:34s}{detail}")
    failed = sum(1 for _, ok, _ in rows if not ok)
    print(f"{len(rows) - failed} passed, {failed} failed")

    checks_payload = [{"label": label, "ok": bool(ok), "detail": detail} for label, ok, detail in rows]
    print("@@RESULT " + json.dumps({
        "file": out.name,
        "failed": failed,
        "params": p,
        "checks": checks_payload,
        "echo": echo_info,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
