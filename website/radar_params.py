"""Pure-Python parameter handling (no MyHDL, no NumPy), shared by app.py and runner.py.

The timing engine runs on a 100 MHz clock (10 ns period). PW and PRI are therefore
whole numbers of clock cycles, and the engine registers hold "count max" values:

    pwCountMax  = PW cycles  - 1      (Mode 1: 8 cycles = 80 ns   -> 7)
    prfCountMax = PRI cycles - 1      (Mode 1: 33333 cycles       -> 33332)
"""

CLK_NS = 10
C = 299_792_458.0                # speed of light, m/s

# Hardware limits taken from the intbv ranges in pwCrtl.py / prfEngine.py / rxBlanking.py
MAX_PW_CYCLES = 200
MAX_PRI_CYCLES = 200_000
MAX_GUARD_CYCLES = 100
MAX_PULSES = 10

# Same carrier window as rfwave.py: 9410 MHz +/- 30 MHz
FC_NOM_MHZ = 9410.0
FC_TOL_MHZ = 30.0


class ParamError(ValueError):
    """Raised when a requested parameter set cannot be built in hardware."""


def _num(value, name):
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ParamError(f"{name} must be a number")


def resolve(pw_ns, pri_us, pulses=3, guard_cycles=5, fc_mhz=FC_NOM_MHZ):
    pw_ns = _num(pw_ns, "PW")
    pri_us = _num(pri_us, "PRI")
    pulses = int(_num(pulses, "Pulses"))
    guard_cycles = int(_num(guard_cycles, "RX guard"))
    fc_mhz = _num(fc_mhz, "Carrier")

    pw_cycles = int(round(pw_ns / CLK_NS))
    pri_cycles = int(round(pri_us * 1000.0 / CLK_NS))

    if not 1 <= pw_cycles <= MAX_PW_CYCLES:
        raise ParamError(f"PW must be between {CLK_NS} and {MAX_PW_CYCLES * CLK_NS} ns "
                         f"(got {pw_ns:g} ns)")
    if not 0 <= guard_cycles <= MAX_GUARD_CYCLES:
        raise ParamError(f"RX guard must be between 0 and {MAX_GUARD_CYCLES} cycles")
    if not 1 <= pulses <= MAX_PULSES:
        raise ParamError(f"Pulses must be between 1 and {MAX_PULSES}")
    if abs(fc_mhz - FC_NOM_MHZ) > FC_TOL_MHZ:
        raise ParamError(f"Carrier must be {FC_NOM_MHZ:g} +/- {FC_TOL_MHZ:g} MHz")
    if pri_cycles > MAX_PRI_CYCLES:
        raise ParamError(f"PRI must be at most {MAX_PRI_CYCLES * CLK_NS / 1000:g} us "
                         f"(got {pri_us:g} us)")
    min_pri = pw_cycles + guard_cycles + 4
    if pri_cycles < min_pri:
        raise ParamError(f"PRI is too short: it must cover PW + RX guard + margin, "
                         f"at least {min_pri * CLK_NS / 1000:g} us for these settings")

    pw_actual = pw_cycles * CLK_NS
    pri_actual_ns = pri_cycles * CLK_NS
    return {
        "pw_cycles": pw_cycles,
        "pri_cycles": pri_cycles,
        "pw_count_max": pw_cycles - 1,
        "prf_count_max": pri_cycles - 1,
        "guard_cycles": guard_cycles,
        "pulses": pulses,
        "fc_hz": fc_mhz * 1e6,
        "pw_ns": pw_actual,
        "pri_us": pri_actual_ns / 1000.0,
        "prf_hz": 1e9 / pri_actual_ns,
        "delta_r_m": C * pw_actual * 1e-9 / 2.0,
        "range_km": C * pri_actual_ns * 1e-9 / 2.0 / 1000.0,
        "requested_pw_ns": pw_ns,
        "requested_pri_us": pri_us,
    }
