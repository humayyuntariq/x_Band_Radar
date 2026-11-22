from myhdl import block, always_seq, always_comb, Signal, intbv

@block
def prfEngine (clk, rst, prfCountMax, txTrigger, startCycles=1):
    prfCounter = Signal(intbv(0, min=0, max=200001))
    activePrfMax = Signal(intbv(0, min=0, max=200001))
    triggerInt = Signal(bool(0))
    first = Signal(bool(1))

    @always_seq(clk.posedge, reset=rst)
    def prfProc():
        triggerInt.next = 0
        if first:
            if prfCounter >= startCycles-1:
                first.next = 0
                prfCounter.next = 0
                triggerInt.next = 1
                activePrfMax.next = prfCountMax
            else:
                prfCounter.next = prfCounter + 1
        
        elif prfCounter >= activePrfMax:
             prfCounter.next = 0
             triggerInt.next = 1
             activePrfMax.next = prfCountMax
        else:
            prfCounter.next = prfCounter + 1

    @always_comb
    def drive():
        txTrigger.next = triggerInt

    return prfProc, drive 