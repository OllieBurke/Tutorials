import numpy as np 
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from LISA_utils import FFT, waveform

from plotting_code import (waveform_plot,matched_filter_plot,
                           trace_plot_before_burnin,trace_plot_after_burnin,
                           corner_plot_after_burnin)


def llike(data_f, signal_f, variance_noise_f):
    """
    Computes log likelihood 
    Assumption: Known PSD otherwise need additional term
    Inputs:
    data in frequency domain 
    Proposed signal in frequency domain
    Variance of noise
    """
    inn_prod = sum((abs(data_f - signal_f)**2) / variance_noise_f)
    # print(inn_prod)
    return(-0.5 * inn_prod)


def lprior_uniform(param,param_low_val,param_high_val):
    """
    Set uniform priors on parameters with select ranges.
    """
    if param < param_low_val or param > param_high_val:
        return -np.inf
    else:
        return 0

def lpost(data_f,signal_f, variance_noise_f,param1,param2,param3, param1_low_range = -10,param1_high_range = 10,
                                                   param2_low_range = -10,param2_high_range = 10,
                                                    param3_low_range = -10,param3_high_range = 10):
    '''
    Compute log posterior - require log likelihood and log prior.
    '''
    return(lprior_uniform(param1,param1_low_range,param1_high_range) + 
                lprior_uniform(param2,param2_low_range,param2_high_range) + 
                    lprior_uniform(param3,param3_low_range,param3_high_range) + llike(data_f,signal_f,variance_noise_f))


def accept_reject(lp_prop, lp_prev):
    '''
    Compute log acceptance probability (minimum of 0 and log acceptance rate)
    Decide whether to accept (1) or reject (0)
    '''
    u = np.random.uniform(size = 1)  # U[0, 1]
    logalpha = np.minimum(0, lp_prop - lp_prev)  # log acceptance probability
    if np.log(u) < logalpha:
        return(1)  # Accept
    else:
        return(0)  # Reject

def MCMC_run(data_f, t, variance_noise_f,
                   Ntotal, burnin, param_start, true_vals, Generate_Movies, printerval, save_interval,
                   a_var_prop, f_var_prop, fdot_var_prop):
    '''
    Metropolis MCMC sampler
    '''
    
    plot_direc = os.getcwd() + "/plots"
    # Set starting values

    a_chain = [param_start[0]]
    f_chain = [param_start[1]]
    fdot_chain = [param_start[2]]

    # Initial signal
    
    signal_init_t = waveform(a_chain[0],f_chain[0],fdot_chain[0],t)   # Initial time domain signal
    signal_init_f = FFT(signal_init_t)  # Intial frequency domain signal

    # for plots -- uncomment if you don't care.

    params =[r"$\log_{10}(a)$", r"$\log_{10}(f)$", r"$\log_{10}(\dot{f})$"] 
    N_param = len(params)
    t_hour = t/60/60
    np.random.seed(1234)
    noise_t_plot = np.random.normal(0,8e-21,len(signal_init_t))
    waveform_true_f = FFT(waveform(true_vals[0],true_vals[1],true_vals[2],t))
    matched_filter_vec = []
    opt_SNR = np.sqrt(sum(abs(FFT(waveform(true_vals[0],true_vals[1],true_vals[2],t)))**2 / variance_noise_f ))
    signal_prop_f = signal_init_f

    j1,j2,j3,j4,j5 = 0,0,0,0,0

    # end commented code here if you don't care about plotting.



                                            
    # Initial value for log posterior
    lp = []
    lp.append(lpost(data_f, signal_init_f, variance_noise_f, a_chain[0], f_chain[0],fdot_chain[0]))  # Append first value of log posterior
    
    lp_store = lp[0]  # Create log posterior storage to be overwritten
                 
    #####                                                  
    # Run MCMC
    #####
    accept_reject_count = [1]

    for i in tqdm(range(1, Ntotal)):
        
        if i % printerval == 0: # Print accept/reject ratio.
            accept_reject_ratio = sum(accept_reject_count)/len(accept_reject_count)
            tqdm.write("Iteration {0}, accept_reject = {1}".format(i,accept_reject_ratio))

        if Generate_Movies:
            norm = np.sqrt(sum((abs(signal_prop_f)**2) / variance_noise_f))
            matched_filter = (1/norm) * np.real(sum(np.conjugate(signal_prop_f)*data_f / variance_noise_f))
            matched_filter_vec.append(matched_filter)
            if i % save_interval == 0:
                if i <= burnin:
                    j1 = waveform_plot(j1, t, t_hour, true_vals, a_prop, f_prop, fdot_prop,noise_t_plot, dir = plot_direc)  # Save still images of waveforms
                    j2 = matched_filter_plot(j2, matched_filter_vec, opt_SNR, burnin, dir = plot_direc)                     # Save still images of matched filter
                    j3 = trace_plot_before_burnin(j3,a_chain,f_chain,fdot_chain,true_vals,Ntotal,burnin, dir = plot_direc)  # Save still images of trace plot
                else:
                    j4 = trace_plot_after_burnin(j4,a_chain,f_chain,fdot_chain,true_vals,Ntotal,burnin, dir = plot_direc)   # Save still images of trace plot before burnin
                    j5 = corner_plot_after_burnin(j5, true_vals,a_chain,f_chain,fdot_chain,burnin,params,
                                                a_prop,f_prop,fdot_prop,N_param,dir = plot_direc)                        # Save still images ofcorner plot after burnin 

        lp_prev = lp_store  # Call previous stored log posterior
        
        # Propose new points according to a normal proposal distribution of fixed variance 

        a_prop = a_chain[i - 1] + np.random.normal(0, np.sqrt(a_var_prop))
        f_prop = f_chain[i - 1] + np.random.normal(0, np.sqrt(f_var_prop))
        fdot_prop = fdot_chain[i - 1] + np.random.normal(0, np.sqrt(fdot_var_prop))

        # Propose a new signal      
        signal_prop_t = waveform(a_prop,f_prop,fdot_prop,t)
        signal_prop_f = FFT(signal_prop_t)

        
        # Compute log posterior
        lp_prop = lpost(data_f,signal_prop_f, variance_noise_f,
                        a_prop, f_prop, fdot_prop)
        
        ####
        # Perform accept_reject call
        ####
        # breakpoint()
        if accept_reject(lp_prop, lp_prev) == 1:  # Accept
            f_chain.append(f_prop)    # accept a_{prop} as new sample
            a_chain.append(a_prop)    # accept f_{prop} as new sample
            fdot_chain.append(fdot_prop)    # accept \dot{f}_{prop} as new sample
            accept_reject_count.append(1)
            lp_store = lp_prop  # Overwrite lp_store
        else:  # Reject, if this is the case we use previously accepted values
            a_chain.append(a_chain[i - 1])  
            f_chain.append(f_chain[i - 1])  
            fdot_chain.append(fdot_chain[i - 1])
            accept_reject_count.append(0)

        lp.append(lp_store)
    
    # Recast as .nparrays

    a_chain = np.array(a_chain)
    f_chain = np.array(f_chain)
    fdot_chain = np.array(fdot_chain)

    
    return a_chain,f_chain, fdot_chain,lp  # Return chains and log posterior.