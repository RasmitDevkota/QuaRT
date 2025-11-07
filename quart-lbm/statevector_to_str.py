import numpy

# Numbers below this threshold will be printed as 0
ABSOLUTE_TOLERANCE: float = 1e-6

def _real2str(num: float, decimals: int, atol: float, force_ones: bool) -> str:
    ret = ""
    float_format = "{0:." + str(decimals) + "f}"
    if force_ones or abs(num - 1) > max(10 ** (-decimals) / 2, atol):
        ret += float_format.format(num)
    return ret


def _complex2str(num: complex, decimals: int, atol: float = ABSOLUTE_TOLERANCE) -> str:
    ret = ""
    real, imag = abs(num.real), abs(num.imag)
    if real > atol:
        ret += _real2str(num.real, decimals, atol, force_ones=imag > atol)
        if imag > atol:
            ret += "+" if num.imag > 0 else "-"
    if imag > atol:
        ret += "i" + _real2str(imag, decimals, atol, force_ones=False)
    if real > atol and imag > atol:
        ret = "(" + ret + ")"
    return ret


def _num2str(num, decimals: int, atol: float = 1e-7) -> str:
    return _complex2str(complex(num), decimals, atol=atol)

def bin2ket(num: int, length: int) -> str:
    return "|{state_bin}〉".format(state_bin=bin(num)[2:].zfill(length))

def statevector_to_str(
    statevector: numpy.ndarray, decimals: int = 2, atol: float = ABSOLUTE_TOLERANCE
) -> str:
    ret = ""
    n = numpy.ceil(numpy.log2(statevector.size)).astype(int)
    for i in range(len(statevector)):
        if abs(statevector[i]) > 10 ** (-decimals) / 2:
            ret += "{coeff}{ket} + ".format(
                coeff=_num2str(statevector[i], decimals, atol), ket=bin2ket(i, n)
            )
    return ret[:-3]

