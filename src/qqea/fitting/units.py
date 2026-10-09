"""Unit conversions: RF power/voltage (50 ohm) and population <-> <Z>."""

import numpy as np


def Vp2dBm(Vp, Z=50):
    return 10*np.log10(Vp**2 / (2*Z*1e-3))
def dBm2Vp(dBm, Z=50):
    return np.sqrt(10**(dBm/10)*2*Z*1e-3)
def W2dBm(W):
    return 10 * np.log10(W) + 30
def dBm2W(dBm):
    return 10 ** ((dBm - 30) / 10)

def Z2P(Z):
    return (1 - Z) / 2
def P2Z(P):
    return 1 - 2*P
