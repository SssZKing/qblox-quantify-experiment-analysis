# Author: ZHUO SHEN; March 2024

# Customized S-parameter fitting module for superconducting notch-type resonators.
# Diameter Correcting Method is from https://doi.org/10.1063/1.3692073
# This module is simplified and modified from Martin Ritter's code to incorporate with scikit-rf.

import numpy as np
import skrf as rf
from scipy.optimize import curve_fit

def remove_delay(ntwk, res_type, L=0, R=-1, S='s21'):
    """
    Remove electrical delay from VNA S21 measurement

    Parameters:
        ntwk (skrf.Network) - network instance with full 2-port measurement
        res_type (str) - 'absorption' or 'transmission'
        L (int) - the left point of S21 to determine the deley
        R (int) - the right point of S21 to determine the deley
        S (str) - the S trace needs to be dealed with. Defult: 's21'

    Returns:
        SXX_new (skrf.Network) - 1-port of S-parameter trace.
    """
    SXX = getattr(ntwk, S)
    
    I = SXX.s_re.reshape(-1)
    Q = SXX.s_im.reshape(-1)
    s = I + 1j*Q
    
    phase = SXX.s_rad_unwrap.reshape(-1)
    freq = SXX.f
    slope,_ = np.polyfit(freq[L:R], phase[L:R], 1)
    
    s_new = s*np.exp(-1j*slope*freq)
    I0 = s_new[0].real
    Q0 = s_new[0].imag
    if res_type == 'absorption':
        s_new *= np.exp(-1j*(np.arctan2(Q0, I0)))
    elif res_type == 'transmission':
        s_new *= np.exp(-1j*(np.arctan2(Q0, I0)-np.pi/2))
        
    SXX_new = rf.Network(name = S+'_removed', frequency=freq, s=s_new)
    
    return SXX_new

def DCM_fit(S21_ntwk):
    """
    Fit 1-port network after removing electrical delay with Diameter Correction Method.

    Parameters:
        S21_ntwk (skrf.Network) - network instance with 1-port trace

    Return:
        DCM_fit_ntwk (skrf.Network) - fitted 1-port network
        parameters (dict) - fitted ['f0', 'Q_tot', 'Q_ext', 'amp', 'phi', 'tau', 'phase']
    """
    def base(f, f0, Q_tot, Q_ext):
        return (Q_tot/Q_ext)/(1+2j*Q_tot*(f-f0)/f0)

    # DCM model
    def DCM(f, f0, Q_tot, Q_ext, amp, phi, tau, phase):
        return amp*np.exp(1j*((f-f[0])*tau+phase))*(1-base(f, f0, Q_tot, Q_ext)*np.exp(1j*phi))

    def stack_function(freq, function, *params):
        N = len(freq)
        x = freq[N//2:]
        y = function(x, *params)
        return np.hstack([np.real(y), np.imag(y)])

    def DCM_stack(f, f0, Q_tot, Q_ext, amp, phi, tau, phase):
        return stack_function(f, DCM, f0, Q_tot, Q_ext, amp, phi, tau, phase)
    
    I = S21_ntwk.s_re.reshape(-1)
    Q = S21_ntwk.s_im.reshape(-1)

    freq = S21_ntwk.f
    amp = np.abs(I+1j*Q)
    phase = np.arctan2(Q[0], I[0])
    g = (1-min(amp))/min(amp)
    
    QF = rf.Qfactor(S21_ntwk, res_type='absorption')
    QF.fit()
    Q_tot = QF.Q_L
    Q_int = (1+g)*Q_tot
    Q_ext = 1/(1/Q_tot-1/Q_int)
    f0 = QF.f_L
    
    phi = 0
    tau = 0
    guess  = [f0       , Q_tot , Q_ext , max(amp), phi   , tau    , phase ]
    bounds = ([freq[0] , 10    , 10    , 0       , -np.pi, -np.inf, -np.pi],
              [freq[-1], np.inf, np.inf, max(amp),  np.pi, np.inf ,  np.pi])
    
    N = len(freq)
    sigma_real = np.std(I[0:N//10])*np.ones(N)
    sigma_imag = np.std(Q[0:N//10])*np.ones(N)
    sigma_full = np.hstack([sigma_real, sigma_imag])
    
    popt, pcov = curve_fit(DCM_stack, 
                    np.hstack([freq, freq]), np.hstack([I, Q]), sigma=sigma_full, 
                    p0=guess, bounds=bounds, method='trf')
    
    DCM_fit_ntwk = rf.Network(name='DCM_fit', frequency=freq, s=DCM(freq, *popt))
    
    param_names = ['f0', 'Q_tot', 'Q_ext', 'amp', 'phi', 'tau', 'phase']
    parameters = dict(zip(param_names, popt))
    parameters['Q_ext'] /= np.cos(parameters['phi'])
    parameters['Q_int'] = (1/parameters['Q_tot']-1/parameters['Q_ext'])**-1
    
    return DCM_fit_ntwk, parameters
