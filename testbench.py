from myhdl import (block, always, instance, delay, now,
                   Signal, intbv, ResetSignal, StopSimulation)
from twg import twg
import rfwave
import spectral
import radar_sim

clkns       = 10
pulse       = 3
startCycles = 5
rstCycles   = 2
guardCycles = 5
fc          = 9.41e9
VCD_FILE    = "simulationOrg"

def cycles(n):
    return n * clkns


@block
def testbench(gates, blanks):
    clk      = Signal(bool(0))
    rst      = ResetSignal(0, active=0, isasync=False)
    mode     = Signal(intbv(0)[3:])
    txGate   = Signal(bool(0))
    rxBlank  = Signal(bool(0))
    rxEnable = Signal(bool(0))
    cur      = {"mode": 0}

    dut = twg(clk, rst, mode, txGate, rxBlank, rxEnable,
              guardCycles=guardCycles, startCycles=startCycles)

    @always(delay(clkns // 2))
    def clkgen():
        clk.next = not clk

    @instance
    def stim():
        modes = (0b000, 0b001, 0b010, 0b011, 0b100)
        yield clk.negedge
        cur["mode"] = modes[0]
        rst.next = 0
        mode.next = modes[0]
        yield delay(cycles(rstCycles))
        rst.next = 1

        for m in modes:
            if m != modes[0]:
                yield clk.negedge
                cur["mode"] = m
                mode.next = m
            for i in range(pulse):
                yield txGate.posedge
                print(f"mode {m:03b} pulse {i+1} at {now()} ns")
            yield txGate.negedge
            yield delay(cycles(10))
        yield delay(cycles(124900))
        raise StopSimulation()

    @instance
    def gate_mon():
        while True:
            yield txGate.posedge
            m, r = cur["mode"], now()
            yield txGate.negedge
            gates.append((m, r, now()))

    @instance
    def blank_mon():
        while True:
            yield rxBlank.posedge
            m, r = cur["mode"], now()
            yield rxBlank.negedge
            blanks.append((m, r, now()))

    @always(clk.negedge)
    def chk():
        assert rxEnable != rxBlank, f"rxEnable and rxBlank equal at {now()} ns"

    return dut, clkgen, stim, gate_mon, blank_mon, chk


if __name__ == "__main__":
    gates, blanks = [], []
    tb = testbench(gates, blanks)
    tb.config_sim(trace=True, filename=VCD_FILE, tracebackup=False, timescale="1ns")
    tb.run_sim(duration=20_000_000, quiet=1)
    tb.quit_sim()

    pulses = [dict(mode=g[0], g_rise=g[1], g_fall=g[2], b_rise=b[1], b_fall=b[2])
              for g, b in zip(gates, blanks)]

    rfwave.verify(pulses, fc, guard_ns=guardCycles * clkns)
    res, echo_ev = radar_sim.run_all(pulses, fc)           # prints the table, saves plots/echo/*.png
    rfwave.add_rf_to_vcd(VCD_FILE + ".vcd", pulses, fc, echo_ev)
    #spectral analysis driven by the pulse widths measured from the timing simulation 
    pws = {s["name"]: spectral.pw_from_pulses(pulses, c) for c, s in rfwave.SPECS.items()}
    spectral.run(pws, fc)