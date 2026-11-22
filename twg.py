from myhdl import block, Signal, intbv, always_comb
from rangeSelector import rangeSelector
from prfEngine import prfEngine
from pwCrtl import pwCrtl
from rxBlanking import rxBlanking

@block 
def twg(clk, rst, mode, txGate, rxBlank, rxEnable,guardCycles=10, startCycles=1):
    prfCountMax = Signal(intbv(0, min=0, max=200001))
    pwCountMax = Signal(intbv(0, min=0, max=201))
    txTrigger = Signal(bool(0))
    txGate2 = Signal(bool(0))

    uSel = rangeSelector(mode, prfCountMax, pwCountMax)
    uprf = prfEngine(clk, rst, prfCountMax, txTrigger)
    upw = pwCrtl(clk, rst, txTrigger, pwCountMax, txGate2)
    urxb = rxBlanking(clk, rst, txTrigger, txGate2, rxBlank, rxEnable, guardCycles)

    @always_comb
    def drive():
        txGate.next = txGate2

    return uSel, uprf, upw, urxb, drive