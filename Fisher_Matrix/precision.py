import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np


# set random seed
np.random.seed(42)

from LISA_utils import PowerSpectralDensity, FFT, freq_PSD, inner_prod, waveform

np.random.seed(1234)

def waveform(params):
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

def waveform_deriv(params,phi):
    """
    This is a function. It takes in a value of the amplitude $a$, frequency $f$ and frequency derivative $\dot{f}
    and a time vector $t$ and spits out whatever is in the return function. Modify amplitude to improve SNR. 
    Modify frequency range to also affect SNR but also to see if frequencies of the signal are important 
    for the windowing method. We aim to estimate the parameters $a$, $f$ and $\dot{f}$.
    """
    a = params[0]
    f = params[1]
    fdot = params[2]

    return (a *(np.sin((2*np.pi)*(f*t + 0.5*fdot * t**2) + phi)))

def llike(params, data_f, variance_noise_f):
    """
    Computes log likelihood 
    Assumption: Known PSD otherwise need additional term
    Inputs:
    data in frequency domain 
    Proposed signal in frequency domain
    Variance of noise
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
N_params = len(true_params)

tmax =  120*60*60                 # Final time
fs = 2*f_true                     # Sampling rate
delta_t = np.floor(0.01/fs)       # Sampling interval

t = np.arange(0,tmax,delta_t)     # Form time vector from t0 = 0 to t_{n-1} = tmax. Length N [include zero]

N_t = int(2**(np.ceil(np.log2(len(t)))))   # Round length of time series to a power of two. 
                                           # Length of time series

h_true_t = waveform(true_params)

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

# Compute FM

exact_deriv_a = (a_true)** -1  * waveform_deriv(true_params, 0)
exact_deriv_a_fft = FFT(exact_deriv_a)

exact_deriv_f = (2*np.pi*t) * waveform_deriv(true_params, np.pi/2)
exact_deriv_f_fft = FFT(exact_deriv_f)

exact_deriv_fdot = (0.5) * (2*np.pi * t**2) * waveform_deriv(true_params, np.pi/2)
exact_deriv_fdot_fft = FFT(exact_deriv_fdot)

deriv_vec = [exact_deriv_a_fft, exact_deriv_f_fft, exact_deriv_fdot_fft]
Fisher_Matrix = np.eye(N_params)
for i in range(N_params):
    for j in range(N_params):
        Fisher_Matrix[i,j] = inner_prod(deriv_vec[i],deriv_vec[j],PSD,delta_t,N_t)

Cov_Matrix = np.linalg.inv(Fisher_Matrix)

precision = np.sqrt(np.diag(Cov_Matrix))
# Investigation

eig,eig_vec = np.linalg.eig(Cov_Matrix)

def elipse(center,sd_x, sd_y, scale):
    u = center[0]
    v = center[1]
    a = sd_x
    b = sd_y
    s = scale

    t = np.linspace(0, 2*np.pi, 100)
    x = u+a*s*np.cos(t) 
    y = v+b*s*np.sin(t) 
    plt.plot(x,y)
    plt.show()

elipse([true_params[0],true_params[1]],precision[0],precision[1],5.991)



# from math import pi, cos, sin

# u=1.       #x-position of the center
# v=0.5      #y-position of the center
# a=2.       #radius on the x-axis
# b=1.5      #radius on the y-axis
# t_rot=pi/4 #rotation angle

# t = np.linspace(0, 2*pi, 100)
# Ell = np.array([a*np.cos(t) , b*np.sin(t)])  
#      #u,v removed to keep the same center location
# R_rot = np.array([[cos(t_rot) , -sin(t_rot)],[sin(t_rot) , cos(t_rot)]])  
#      #2-D rotation matrix

# Ell_rot = np.zeros((2,Ell.shape[1]))
# for i in range(Ell.shape[1]):
#     Ell_rot[:,i] = np.dot(R_rot,Ell[:,i])

# plt.plot( u+Ell[0,:] , v+Ell[1,:] )     #initial ellipse
# plt.plot( u+Ell_rot[0,:] , v+Ell_rot[1,:],'darkorange' )    #rotated ellipse
# plt.grid(color='lightgray',linestyle='--')
# plt.show()

