from myhdl import block, enum, Signal, intbv, always_comb, always_seq

@block
def rxBlanking(clk, rst, txTrigger, txGate, rxBlank, rxEnable, guardCycles=10):
    t_state = enum('IDLE', 'BLANKING', 'GUARD')
    state = Signal(t_state.IDLE)
    guardCounter = Signal(intbv(0, min=0, max=101))
    blankInt = Signal(bool(0))
    enInt = Signal(bool(1))                  # reset value 1
    txGateHigh = Signal(bool(0))

    @always_seq(clk.posedge, reset=rst)
    def blankProc():
        if state == t_state.IDLE:
            blankInt.next = 0
            guardCounter.next = 0
            txGateHigh.next = 0
            if txTrigger:
                state.next = t_state.BLANKING
                blankInt.next = 1
                enInt.next = 0               # receiver off with the blank
        elif state == t_state.BLANKING:
            blankInt.next = 1
            if txGate:
                txGateHigh.next = 1
            if (not txGate) and txGateHigh:
                if guardCycles <= 0:
                    state.next = t_state.IDLE
                    blankInt.next = 0
                    enInt.next = 1
                else:
                    state.next = t_state.GUARD
                    guardCounter.next = 0
        elif state == t_state.GUARD:
            blankInt.next = 1
            if guardCycles <= 1 or guardCounter >= guardCycles - 1:
                state.next = t_state.IDLE
                blankInt.next = 0
                enInt.next = 1               # receiver on, same edge as blank falls
            else:
                guardCounter.next = guardCounter + 1

    @always_comb
    def drive():
        rxBlank.next = blankInt
        rxEnable.next = enInt

    return blankProc, drive