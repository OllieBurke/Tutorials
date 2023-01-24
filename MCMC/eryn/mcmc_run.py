from eryn.ensemble import EnsembleSampler
from eryn.ensemble import EnsembleSampler
from eryn.state import State
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.utils import TransformContainer
from eryn.moves import StretchMove
from eryn.backends import HDFBackend
from eryn.utils.utility import groups_from_inds

import matplotlib.pyplot as plt
import numpy as np

# set random seed
np.random.seed(42)

from LISA_utils import PowerSpectralDensity, FFT, freq_PSD, inner_prod, waveform

np.random.seed(1234)

def waveform_check(params):
    """
    This is a function. It takes in a value of the amplitude $a$, frequency $f$ and frequency derivative $\dot{f}
    and a time vector $t$ and spits out whatever is in the return function. Modify amplitude to improve SNR. 
    Modify frequency range to also affect SNR but also to see if frequencies of the signal are important 
    for the windowing method. We aim to estimate the parameters $a$, $f$ and $\dot{f}$.
    """
    a = params[0]
    f = params[1]
    fdot = params[2]

    return (a *(np.sin((2*np.pi)*(f*t + 0.5*fdot * t**2))))

def llike(params, data_f, variance_noise_f):
    """
    Computes log likelihood 
    Assumption: Known PSD otherwise need additional term
    Inputs:
    data in frequency domain 
    Proposed signal in frequency domain
    Variance of noise
    """

    signal_prop_t = waveform_check(params)
    signal_prop_f = FFT(signal_prop_t)

    inn_prod = sum((abs(data_f - signal_prop_f)**2) / variance_noise_f)

    return(-0.5 * inn_prod)

# Set true parameters

a_true = 5e-21
f_true = 1e-3
fdot_true = 1e-8  

true_params = [a_true,f_true,fdot_true]

tmax =  120*60*60                 # Final time
fs = 2*f_true                     # Sampling rate
delta_t = np.floor(0.01/fs)       # Sampling interval

t = np.arange(0,tmax,delta_t)     # Form time vector from t0 = 0 to t_{n-1} = tmax. Length N [include zero]

N_t = int(2**(np.ceil(np.log2(len(t)))))   # Round length of time series to a power of two. 
                                           # Length of time series

h_true_t = waveform_check(true_params)

h_true_f = FFT(h_true_t)         # Compute true signal in
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

data_f = h_true_f + 0*noise_f         # Construct data stream

llike_kwargs = {"T_obs":t, "data_stream":data_f, "variance_noise_f":variance_noise_f}

ndim = 3
nwalkers = 100

delta_a = np.sqrt(7.8152977583191198e-46)
delta_f = np.sqrt(3.122370011848878e-17)
delta_dotf = np.sqrt(1.007508992696005e-27)


start_a = true_params[0]*(1 + delta_a/true_params[0] * np.random.uniform(1,2,size = nwalkers).reshape(nwalkers,1))
start_f = true_params[1]*(1 + delta_f/true_params[1] * np.random.uniform(1,2,size = nwalkers).reshape(nwalkers,1))
start_fdot = true_params[2]*(1 + delta_dotf/true_params[2] * np.random.uniform(1,2,size = nwalkers).reshape(nwalkers,1))

start_params = np.hstack([start_a,start_f,start_fdot])


priors_in = {
    0: uniform_dist(5e-23, 5e-19),
    1: uniform_dist(1e-5, 1e-2),
    2: uniform_dist(1e-11, 1e-5)
}

priors = ProbDistContainer(priors_in)


from multiprocessing import (get_context,Pool,cpu_count)
pool = get_context("fork").Pool(10)


file_name = "sample_parameters"
backend = HDFBackend(file_name)
# start = backend.get_last_sample()

breakpoint()
# backend.reset(nwalkers, ndim)

ensemble = EnsembleSampler(
    nwalkers,
    ndim,
    llike,
    priors,
    args=[data_f, variance_noise_f],
    pool = pool,
    backend = backend
)

nsteps = 500
# burn for 100 steps
burn = 0
# thin by 5
thin_by = 5

out = ensemble.run_mcmc(start_params, nsteps, burn=burn, progress=True, thin_by=thin_by)


breakpoint()


# MCMC - parameter estimation

