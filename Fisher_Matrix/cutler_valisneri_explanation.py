import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
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

def deriv_waveform(params,phi):
    """
    This is a function. It takes in a value of the amplitude $a$, frequency $f$ and frequency derivative $\dot{f}
    and a time vector $t$ and spits out whatever is in the return function. Modify amplitude to improve SNR. 
    Modify frequency range to also affect SNR but also to see if frequencies of the signal are important 
    for the windowing method. We aim to estimate the parameters $a$, $f$ and $\dot{f}$.
    """
    a = params[0]
    f = params[1]
    fdot = params[2]

    return (a *(np.sin((2*np.pi)*(f*t + 0.5*fdot * t**2) + phi) ))

def Gaussian(values,mean,std):
    return np.exp(-(values - mean)**2 / (2*std**2))

def confidence_ellipse(cov, center, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])

    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = center[0]

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = center[1]

    eig,vec = np.linalg.eig(cov)  # Eigenvalues are the standard deviations

    vec_0 = vec[:,0] ; vec_1 = vec[:,1]
    alpha = np.arctan2(np.linalg.norm(vec_0),np.linalg.norm(vec_1))*(180/np.pi)  # Convert to degrees
    transf = transforms.Affine2D() \
        .rotate_deg(alpha) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

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

# Compute FM

exact_deriv_a = (a_true)** -1  * deriv_waveform(true_params, 0)
deriv_a_fft = FFT(exact_deriv_a)

exact_deriv_f = (2*np.pi*t) * deriv_waveform(true_params, np.pi/2)
deriv_f_fft = FFT(exact_deriv_f)

exact_deriv_fdot = (0.5) * (2*np.pi * t**2) * deriv_waveform(true_params, np.pi/2)
deriv_fdot_fft = FFT(exact_deriv_fdot)

deriv_vec = [deriv_a_fft, deriv_f_fft, deriv_fdot_fft]
Fisher_Matrix = np.eye(N_params)
for i in range(N_params):
    for j in range(N_params):
        Fisher_Matrix[i,j] = inner_prod(deriv_vec[i],deriv_vec[j],PSD,delta_t,N_t)

Cov_Matrix = np.linalg.inv(Fisher_Matrix)

precision = np.sqrt(np.diag(Cov_Matrix))

# Now compute Cutler valisneri biases due to noise 

variance_noise_f = N_t * PSD / (4 * delta_t)            # Calculate variance of noise, real and imaginary.
N_f = len(variance_noise_f)                             # Length of signal in frequency domain

noise_bias_f_list = []
from tqdm import tqdm as tqdm
for j in tqdm(range(0,500)):
    noise_f_seed = np.random.normal(0,np.sqrt(variance_noise_f),N_f) + 1j*np.random.normal(0,np.sqrt(variance_noise_f),N_f)

    b_vec = [inner_prod(noise_f_seed,deriv_vec[i],PSD,delta_t,N_t) for i in range(N_params)]
    noise_bias_f = true_params + Cov_Matrix @ b_vec

    noise_bias_f_list.append(noise_bias_f)

# ----------- Take a random selection of points with FM uncertainty ------------

# Create smooth gaussian distribution for parameter 0
x_range = np.linspace(true_params[0] - 4*np.sqrt(Cov_Matrix[0,0]), 
                      true_params[0] + 4*np.sqrt(Cov_Matrix[0,0]), 1000)

mean_vals = noise_bias_f_list[0:10]
uncertainty_param_0 = precision[0]

list_y_gaussian = Gaussian(x_range, true_params[0], np.sqrt(Cov_Matrix[0,0]))






# Plot histogram of noise biases vs theoretical Gaussian
fig, ax = plt.subplots(figsize=(10, 6))
param_0_values = [noise_bias_f_list[j][0] for j in range(len(noise_bias_f_list))]
hist_counts, _, _ = ax.hist(param_0_values, bins=30, alpha=0.7, density=False, 
                            label='Noise Biases', color='skyblue', edgecolor='black')
max_bin_height = np.max(hist_counts)

# Create smooth gaussian distribution for parameter 0
x_range = np.linspace(true_params[0] - 4*np.sqrt(Cov_Matrix[0,0]), 
                      true_params[0] + 4*np.sqrt(Cov_Matrix[0,0]), 1000)
# Get histogram data to find maximum bin height
y_gaussian = max_bin_height*Gaussian(x_range, true_params[0], np.sqrt(Cov_Matrix[0,0]))


# Extract parameter 0 values from noise bias list

ax.plot(x_range, y_gaussian, label='Theoretical Gaussian', 
    color='red', linewidth=2)

# Add vertical line for true parameter value
ax.axvline(true_params[0], color='green', linestyle='--', 
       linewidth=2, label='True Parameter')

# Formatting
ax.set_xlabel('Parameter 0 Value')
ax.set_ylabel('Density')
ax.set_title('Noise Bias Distribution vs Theoretical Prediction')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

