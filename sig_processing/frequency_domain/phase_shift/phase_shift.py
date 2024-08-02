# Import every package under the sun.

# Created ~ 23:00 on 8th September, 2020.

import numpy as np

from LISA_utils import FFT, freq_PSD, inner_prod, waveform, two_side_pad
import matplotlib.pyplot as plt
from scipy.signal import tukey
from corner import corner
np.random.seed(1234)

a_true = 5e-21
f_true = 1e-3
fdot_true = 1e-8          

tmax =  120*60*60                 # Final time
fs = 2*f_true                     # Sampling rate
delta_t = np.floor(0.01/fs)       # Sampling interval -- largely oversampling here. 

t = np.arange(0,tmax,delta_t)     # Form time vector from t0 = 0 to t_{n-1} = tmax. Length N [include zero]

N_t = int(2**(np.ceil(np.log2(len(t)))))   # Round length of time series to a power of two. 
                                           # Length of time series

h_t = waveform(a_true,f_true,fdot_true,t)
n_t = len(h_t)

window = tukey(n_t,0.1)
h_t_w = window*h_t
h_t_w_2p = two_side_pad(h_t_w)
t_pad = np.arange(0,len(h_t_w_2p)*delta_t,delta_t)

h_2p_f = np.fft.rfft(h_t_w_2p)

freq,PSD = freq_PSD(t,delta_t)  # Extract frequency bins and PSD.

SNR2 = inner_prod(h_2p_f,h_2p_f,PSD,delta_t,N_t)        # Compute optimal matched filtering SNR
print("SNR of source",np.sqrt(SNR2))
variance_noise_f = N_t * PSD / (4 * delta_t)            # Calculate variance of noise, real and imaginary.
N_f = len(variance_noise_f)                             # Length of signal in frequency domain
np.random.seed(1235)                                    # Set the seed

# Now try to prove that multiplication by a complex exponential really induces a shift

td = 2*60*60
time_shift = np.exp(-2*np.pi*1j*freq*td)
h_2p_f_shift = time_shift * h_2p_f

h_t_w_2p_shift = np.fft.irfft(h_2p_f_shift)

pow_2 = np.ceil(np.log2(n_t))
K = int((2**pow_2)-n_t)/2 # Number of zeros we need to pad
K_minus = int(np.floor(K))

plt.plot((t_pad - K_minus *delta_t)/60/60,h_t_w_2p)
plt.plot((t_pad - K_minus * delta_t)/60/60,h_t_w_2p_shift, alpha = 0.5,label = 'shifted')
plt.legend()
plt.show()

# data = np.arange(0,100,1)
# breakpoint()
# N = len(data)
# pow_2 = np.ceil(np.log2(N))
# K = int((2**pow_2)-N)/2 # Number of zeros we need to pad
# K_minus = int(np.floor(K))
# K_pos = int(np.ceil(K))
# # check
# print(K_minus + K_pos - K)
# data_pad = np.pad(data,(K_minus,K_pos),'constant')

# print(len(data_pad))
# print(data_pad)
