# Import every package under the sun.

# Created ~ 23:00 on 8th September, 2020.

import sys
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import signal
import os
import warnings

from LISA_utils import PowerSpectralDensity, FFT, freq_PSD, inner_prod, waveform
from mcmc_fun import MCMC_run

np.random.seed(1234)


# Initial values plus sampling properties

np.random.seed(0)


a_true = 5e-21
f_true = 1e-3
fdot_true = 1e-8          # Fixed - Can be modfiied to include further parameters to understand correlations

tmax =  120*60*60               # Final time
fs = 2*f_true                     # Sampling rate
delta_t = np.floor(0.01/fs)        # Sampling interval
print('delta_t should be ',delta_t, 'to resolve highest frequency in signal')  # This is what the 
                                                                               # sampling interval should be
t = np.arange(0,tmax,delta_t)   # Form time vector from t0 = 0 to t_{n-1} = tmax. Length N [include zero]

N_t = int(2**(np.ceil(np.log2(len(t)))))   # Round length of time series to a power of two. 
                                           # Length of time series

h_true_f = FFT(waveform(a_true,f_true,fdot_true,t))              # Compute true signal in
                                                                                  # frequency domain

freq,PSD = freq_PSD(t,delta_t)  # Extract frequency bins and PSD.

os.chdir('/Users/oburke/Documents/LISA_Science/Tutorials/Bayesian_Statistics_Tutorial/MCMC/plots/PSD_plot')
# plt.figure(figsize = (12,8))
# plt.loglog(freq,PSD)
# plt.xlabel(r'Frequency [Hz]', fontsize = 20)
# plt.ylabel(r'Power Spectral Density [s]', fontsize = 20)
# plt.title('Plot of the PSD', fontsize = 20)
# plt.xlim([1e-4,1e-1])
# plt.ylim([1e-41,1e-32])
# plt.xticks(fontsize= 16)
# plt.yticks(fontsize = 16)
# plt.tight_layout()
# plt.savefig("PSD_Plot.pdf")
# plt.clf()

# plt.figure(figsize = (12,8))
# plt.loglog(freq,2*np.sqrt(freq*abs(delta_t*h_true_f)**2),label = r'$2f\cdot |\hat{h}(f)|$')
# plt.loglog(freq,np.sqrt(PSD), label = r'$\sqrt{S_{n}(f)}$')
# plt.legend(fontsize = 20)
# plt.xlabel(r'Frequency [Hz]', fontsize = 20)
# plt.ylabel(r'Magnitude - Fourier domain', fontsize = 20)
# plt.title('Comparison between the PSD and signal', fontsize = 20)
# plt.xlim([1e-4,1e-1])
# plt.ylim([1e-30,1e-14])
# plt.grid()
# plt.xticks(fontsize= 16)
# plt.yticks(fontsize = 16)
# plt.tight_layout()
# plt.savefig("Comparison_signal_PSD.pdf")
# plt.show()
# Compute SNR

SNR2 = inner_prod(h_true_f,h_true_f,PSD,delta_t,N_t)
print("SNR of source",np.sqrt(SNR2))
variance_noise_f = N_t * PSD / (4 * delta_t)
N_f = len(variance_noise_f)
np.random.seed(1235)
noise_f = np.random.normal(0,np.sqrt(variance_noise_f),N_f) + 1j*np.random.normal(0,np.sqrt(variance_noise_f),N_f) 

data_f = h_true_f + noise_f

# Do the MCMC

Ntotal = 30000  # Total number of iterations
burnin = 0   # Set burn-in. This is the amount of samples we will discard whilst looking 
                 # for the true parameters

variance_noise_f = N_t * PSD / (4 * delta_t)

delta_a = np.sqrt(7.8152977583191198e-46)
# delta_a = np.sqrt(7.8152977583191198e-5)
delta_f = np.sqrt(3.122370011848878e-17)
delta_dotf = np.sqrt(1.007508992696005e-27)

param_start = [a_true + 1000*delta_a, f_true + 500*delta_f, fdot_true - 500*delta_dotf]
true_vals = [a_true,f_true, fdot_true]

a_chain,f_chain,fdot_chain,lp  = MCMC_run(data_f, t, variance_noise_f,
                            Ntotal, param_start,true_vals,
                            printerval = 500, save_interval = 50, # After how many iterations do we print  
                            a_var_prop = delta_a**2,
                            f_var_prop = delta_f**2,
                            fdot_var_prop = delta_dotf**2)  # Initial variance

breakpoint()
# samples = [a_chain, np.log10(f_chain), np.log10(fdot_chain)]
# true_vals = [a_true,np.log10(f_true), np.log10(fdot_true)]
# param_label = [r'$A$',r'$\log_{10}(f)$',r'$\log_{10}(\dot{f})$']
# color = ['green','black','purple']
# fig,ax = plt.subplots(3,1,figsize = (16,8))
# for i in range(0,3):
#     ax[i].plot(samples[i], color = color[i])
#     ax[i].axhline(true_vals[i])
#     ax[i].set_xlabel('Iteration',fontsize = 20)
#     ax[i].set_ylabel(param_label[i], fontsize = 20)

# plt.tight_layout()
# plt.show()

# f_chain_log = np.log10(f_chain)
# fdot_chain_log = np.log10(fdot_chain)
# # print(np.var(a_chain))
# # print(np.var(f_chain))
# # print(np.var(fdot_chain))

# # plt.plot(a_chain);plt.show()
# # plt.plot(f_chain);
# # plt.axhline(f_chain[0]);plt.show()
# # plt.plot(fdot_chain);plt.show()

# params =[r"$a$", r"$\log_{10}(f)$", r"$\log_{10}(\dot{f})$"] 

# N_param = len(params)

# true_vals = [a_chain[0],np.log10(f_chain[0]),np.log10(fdot_chain[0])]


# import corner
# samples = np.column_stack([a_chain,f_chain_log,fdot_chain_log])
# figure = corner.corner(samples,bins = 30, color = 'blue',plot_datapoints=False,smooth1d=False,
#                        labels=params, 
#                        label_kwargs = {"fontsize":12},set_xlabel = {'fontsize': 20},
#                        show_titles=True, title_fmt='.7f',title_kwargs={"fontsize": 9},smooth = False)

# axes = np.array(figure.axes).reshape((N_param, N_param))
# for i in range(N_param):
#     ax = axes[i, i]
#     ax.axvline(true_vals[i], color="r")
    
# for yi in range(N_param):
#     for xi in range(yi):
#         ax = axes[yi, xi]
#         ax.axhline(true_vals[yi], color="r")
#         ax.axvline(true_vals[xi],color= "r")
#         ax.plot(true_vals[xi], true_vals[yi], "sr")
        
# for ax in figure.get_axes():
#     ax.tick_params(axis='both', labelsize=8)
# plt.tight_layout()
# plt.show()
# plt.hist(f_chain,bins = 30);
# plt.axvline(f_chain[0]);plt.show()
# os.chdir('/Users/oburke/Documents/LISA_Science/Tutorials/Bayesian_Statistics_Tutorial/MCMC/plots/movies')
# fig = plt.hist(np.log10(f_chain),bins = 30)
# plt.xlabel(r'Frequency [Hz]')
# plt.ylabel(r'Posterior')
# plt.savefig("f_chain_hist.png")
# plt.plot(f_chain);plt.show()

# a_burned = a[burnin:]
# lp_burned = lp[burnin:]
# a_vec_samples[n].append(a_burned)
# lp_samples[n].append(lp_burned)