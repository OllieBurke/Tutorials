from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.backends import HDFBackend

from LISA_utils import FFT, freq_PSD, inner_prod

import os
import numpy as np

# Documentation: https://mikekatz04.github.io/Eryn/html/user/prior.html

# set random seed
np.random.seed(1234)

def waveform(params):
    """
    This is a function. It takes in a value of the amplitude $a$, frequency $f$ and frequency derivative $\dot{f}
    and a time vector $t$ and spits out whatever is in the return function.
    We aim to estimate the parameters $a$, $f$ and $\dot{f}$.
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
    data in frequency domain  - data_f
    Variance of noise in frequency domain - variance_noise_f

    outputs: log-likelihood.
    """

    signal_prop_t = waveform(params)
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

t = np.arange(0,tmax,delta_t)     # Form time vector from t0 = 0 to t_{n-1} = tmax. Length N [include t = zero]

N_t = int(2**(np.ceil(np.log2(len(t)))))   # Round length of time series to a power of two. 
                                           # Length of time series

import time
start = time.time()
h_true_t = waveform(true_params)
end = time.time() - start
print("Time taken to evaluate waveform is ",end," seconds")

h_true_f = FFT(h_true_t)                        # Compute true signal in
                                                # frequency domain. Real signal so only considering
                                                # positive frequencies here.

freq,PSD = freq_PSD(t,delta_t)  # Extract frequency bins and PSD.

SNR2 = inner_prod(h_true_f,h_true_f,PSD,delta_t,N_t)    # Compute optimal matched filtering squared signal-to-noise ratio (SNR)
breakpoint()
print("SNR of source",np.sqrt(SNR2))

variance_noise_f = N_t * PSD / (4 * delta_t)            # Calculate variance of noise, real and imaginary.
N_f = len(variance_noise_f)                             # Length of signal in frequency domain
np.random.seed(1235)                                    # Set the seed

# Generate frequency domain noise
noise_f = np.random.normal(0,np.sqrt(variance_noise_f)) + 1j*np.random.normal(0,np.sqrt(variance_noise_f)) 

data_f = h_true_f + 0*noise_f         # Construct data stream - using zero noise for now. 

##################### SET UP SAMPLER #########################

ndim = len(true_params)               # Dimension of parameter space
nwalkers = 100          # Number of walkers used to explore parameter space
ntemps = 2             # Number of temperatures used for parallel tempering scheme.
                       # Each group of walkers (equal to nwalkers) is assigned a temperature from T = 1, ... , ntemps.

tempering_kwargs=dict(ntemps=ntemps)  # Sampler requires the number of temperatures as a dictionary

nsteps = 500                          # Number of saved iterations PER walker. Number of samples = nsteps * nwalkers          
burn = 1000                           # Set burn in. This number will discard "burn" from every walker chain. May need to increase/decrease depending on results.
                                      # For example, in the final set of samples, you will have nwalkers * (nsteps - burn)
# thin by 5
thin_by = 1                           # Pick 1 walker. We save an interation after "thin_by" iterations have been made. 
                                      # Total number of iterations: nwalkers * nsteps * thin_by. We will only save 
                                      # nsteps * nwalkers. 

######################## STARTING COORDINATES ##########################

# If we want to explore local posterior, start extremely close to true parameters. See code below 

# Starting values. Here I know (from previous simulations) roughly what the width of the likelihood is at the true parameters
# We need to assign a (different) starting value to every walker. We generate a starting value, multiply that by a 
# random number \in (1,2) and shape this to so it's a row vector with nwalkers elements. We do this for each parameter.

# delta_a = np.sqrt(7.8152977583191198e-46)
# delta_f = np.sqrt(3.122370011848878e-17)
# delta_dotf = np.sqrt(1.007508992696005e-27)

# start_a = true_params[0]*(1 + delta_a/true_params[0] * np.random.uniform(1,2,size = nwalkers).reshape(nwalkers,1))
# start_f = true_params[1]*(1 + delta_f/true_params[1] * np.random.uniform(1,2,size = nwalkers).reshape(nwalkers,1))
# start_fdot = true_params[2]*(1 + delta_dotf/true_params[2] * np.random.uniform(1,2,size = nwalkers).reshape(nwalkers,1))

# # We stack the starting values to be used with the ensemble sampler
# start_params = np.hstack([start_a,start_f,start_fdot])

# if ntemps > 1:
#     # If we decide to use parallel tempering, we fall into this if statement. We assign each *group* of walkers
#     # an associated temperature. We take the original starting values and "stack" them on top of each other. 

#     # Imagine this: You have a cube with dimensions x, y, z. x represents the number of parameters, y represents
#     # the number of walkers, and z represents the number of temperatures. This simply takes the "start_params"
#     # variable above and stacks it ontop of itself ntemp times. 
#     start_params = np.tile(start_params,(ntemps,1,1))


# If we want to pull starting values from the prior (strictly speaking, more statistically "sound" than what I'm doing above).

# Set priors: 0 corresponds to first parameter, 1 the second and 2 the third etc. 
# Manually chosen lower and upper limits. Widen them and see the results.

priors_in = {
    0: uniform_dist(1e-21, 1e-20),
    1: uniform_dist(5e-4, 5e-3),
    2: uniform_dist(5e-9, 5e-8)
}  

priors = ProbDistContainer(priors_in)   # Set up priors so they can be used with the sampler.

start_params = np.tile(priors.rvs(size = nwalkers),(ntemps,1,1))

from multiprocessing import (get_context,cpu_count)
N_cpus = cpu_count()

breakpoint()
pool = get_context("fork").Pool(N_cpus)        # M1 chip -- allows multiprocessing

os.chdir("samples")  # Change to samples directory

file_name = "sample_parameters_tempering.h5"  # Set name of backend
backend = HDFBackend(file_name)            # Initialise backend

# start_params = backend.get_last_sample()   # Uncomment me if you wish to start the sampler again. 


ensemble = EnsembleSampler(
    nwalkers,          
    ndim,
    llike,
    priors,
    args=[data_f, variance_noise_f],   # Arguments in the likelihood function. These are "constant" and do not change in the likelihood.
    pool = pool,                       # Set up multiprocessing to use all my chains
    backend = backend,                 # Store samples to a .h5 file
    tempering_kwargs=tempering_kwargs  # Allow tempering!
)

out = ensemble.run_mcmc(start_params, nsteps, burn=burn, progress=True)  # Run the sampler

breakpoint()


