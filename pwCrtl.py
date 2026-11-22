from myhdl import block, enum, always_comb, always_seq, Signal, intbv

@block 
def pwCrtl (clk, rst, txTrigger, pwCountMax, txGate):
    t_state = enum('IDLE', 'ACTIVE')
    state = Signal(t_state.IDLE)
    pwCounter = Signal(intbv(0, min=0, max=201))
    activePwMax = Signal(intbv(0, min=0, max=201))
    gateInt = Signal(bool(0))

    @always_seq(clk.posedge, reset=rst)
    def pwProc():
        if state == t_state.IDLE:
            gateInt.next = 0
            pwCounter.next = 0
            if txTrigger:
                state.next = t_state.ACTIVE
                gateInt.next = 1
                activePwMax.next = pwCountMax
        elif state == t_state.ACTIVE:
            gateInt.next = 1
            if pwCounter >= activePwMax:
                state.next = t_state.IDLE
                gateInt.next = 0
                pwCounter.next = 0
            else:
                pwCounter.next = pwCounter + 1
    @always_comb
    def drive():
        txGate.next = gateInt

    return pwProc, drive
