from myhdl import (block, always, instance, delay, now, Signal, intbv, ResetSignal, StopSimulation)
from twg import twg
import rfwave

clkNs       = 10
pulse       = 3
startCycles = 5
rstCycles   = 2
guardCycles = 5
fc          = 9.41e9          # carrier; try 9.45e9 to see the range check fire

def cycles(n):
    return n * clkNs


@block
def testbench(gates, blanks):
    clk     = Signal(bool(0))
    rst     = ResetSignal(0, active=0, isasync=False)
    mode    = Signal(intbv(0)[2:])
    txGate  = Signal(bool(0))
    rxBlank = Signal(bool(0))
    cur     = {"mode": 0}                      

    dut = twg(clk, rst, mode, txGate, rxBlank, guardCycles=guardCycles, startCycles=startCycles)

    @always(delay(clkNs // 2))
    def clkgen():
        clk.next = not clk

    @instance
    def stim():
        for m in (0b00, 0b10):
            yield clk.negedge
            cur["mode"] = m
            rst.next = 0
            mode.next = m
            yield delay(cycles(rstCycles))
            rst.next = 1
            for i in range(pulse):
                yield txGate.posedge
                print(f"mode {m:02b} pulse {i+1} at {now()} ns")
            yield txGate.negedge
            yield delay(cycles(10))
        raise StopSimulation()

    @instance
    def gate_mon():                            # logs (mode, rise ns, fall ns) per pulse
        while True:
            yield txGate.posedge
            r = now()
            yield txGate.negedge
            gates.append((cur["mode"], r, now()))

    @instance
    def blank_mon():
        while True:
            yield rxBlank.posedge
            r = now()
            yield rxBlank.negedge
            blanks.append((cur["mode"], r, now()))

    return dut, clkgen, stim, gate_mon, blank_mon


if __name__ == "__main__":
    gates, blanks = [], []
    tb = testbench(gates, blanks)
    tb.config_sim(trace=True, filename="simulation", tracebackup=False, timescale="1ns")
    tb.run_sim(duration=10_000_000, quiet=1)
    tb.quit_sim()

    pulses = [dict(mode=g[0], g_rise=g[1], g_fall=g[2], b_rise=b[1], b_fall=b[2]) for g, b in zip(gates, blanks)]

    rfwave.write_vcd("rfwaveform.vcd", pulses, fc)
    rfwave.verify(pulses, fc, guard_ns=guardCycles * clkNs)
    rfwave.plot(pulses, fc)