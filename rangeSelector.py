from myhdl import block, always_comb

@block
def rangeSelector (mode, prfCountMax, pwCountMax):
    @always_comb
    def sel():
        if mode == 0b000:
            prfCountMax.next = 33332
            pwCountMax.next = 7
        elif mode == 0b001:
            prfCountMax.next = 166665
            pwCountMax.next = 79
        elif mode == 0b010:
            prfCountMax.next = 49999
            pwCountMax.next = 19
        elif mode == 0b011:
            prfCountMax.next = 99999
            pwCountMax.next = 39
        elif mode == 0b100:
            prfCountMax.next = 124999
            pwCountMax.next = 59
        else:
            prfCountMax.next = 33332
            pwCountMax.next = 7
    return sel

    