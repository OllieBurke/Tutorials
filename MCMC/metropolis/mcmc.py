# Import every package under the sun.

# Created ~ 23:00 on 8th September, 2020.

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from LISA_utils import PowerSpectralDensity, FFT, freq_PSD, inner_prod, waveform
from mcmc_fun import MCMC_run
from plotting_code import plot_PSD

np.random.seed(1234)
np.random.seed(0)

# Set true parameters

a_true = 5e-21
f_true = 1e-3
fdot_true = 1e-8          

tmax =  120*60*60                 # Final time
fs = 2*f_true                     # Sampling rate
delta_t = np.floor(0.01/fs)       # Sampling interval

t = np.arange(0,tmax,delta_t)     # Form time vector from t0 = 0 to t_{n-1} = tmax. Length N [include zero]

N_t = int(2**(np.ceil(np.log2(len(t)))))   # Round length of time series to a power of two. 
                                           # Length of time series

h_true_f = FFT(waveform(a_true,f_true,fdot_true,t))         # Compute true signal in
                                                            # frequency domain. Real signal so only considering
                                                            # positive frequencies here.

freq,PSD = freq_PSD(t,delta_t)  # Extract frequency bins and PSD.

SNR2 = inner_prod(h_true_f,h_true_f,PSD,delta_t,N_t)    # Compute optimal matched filtering SNR
print("SNR of source",np.sqrt(SNR2))
variance_noise_f = N_t * PSD / (4 * delta_t)            # Calculate variance of noise, real and imaginary.
N_f = len(variance_noise_f)                             # Length of signal in frequency domain
np.random.seed(1235)                                    # Set the seed

# Generate frequency domain noise
noise_f = np.random.normal(0,np.sqrt(variance_noise_f),N_f) + 1j*np.random.normal(0,np.sqrt(variance_noise_f),N_f) 

data_f = h_true_f + noise_f         # Construct data stream

# MCMC - parameter estimation

Ntotal = 30000  # Total number of iterations
burnin = 6000   # Set burn-in. This is the amount of samples we will discard whilst looking 
             # for the true parameters

variance_noise_f = N_t * PSD / (4 * delta_t)

delta_a = np.sqrt(7.8152977583191198e-46)
delta_f = np.sqrt(3.122370011848878e-17)
delta_dotf = np.sqrt(1.007508992696005e-27)

param_start = [a_true + 1000*delta_a, f_true + 750*delta_f, fdot_true - 750*delta_dotf]  # Starting values
true_vals = [a_true,f_true, fdot_true]   # True values

a_chain,f_chain,fdot_chain,lp  = MCMC_run(data_f, t, variance_noise_f,
                            Ntotal, burnin, param_start,true_vals,
                            printerval = 500, save_interval = 50, 
                            a_var_prop = delta_a**2,
                            f_var_prop = delta_f**2,
                            fdot_var_prop = delta_dotf**2,
                            Generate_Plots = False)  
breakpoint()   # Set breakpoint to investigate samples for a, f and \dot{f}.

print("delta_a = ", np.sqrt(np.var(a_chain[burnin:])))
print("delta_f = ", np.sqrt(np.var(f_chain[burnin:])))
print("delta_fdot = ", np.sqrt(np.var(fdot_chain[burnin:])))
