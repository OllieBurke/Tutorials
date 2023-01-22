import numpy as np 
import os
import matplotlib.pyplot as plt
from LISA_utils import FFT, waveform
import corner

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
                   Ntotal, param_start, true_vals, printerval, save_interval,
                   a_var_prop, f_var_prop, fdot_var_prop):
    '''
    Metropolis MCMC sampler
    '''
    
    # Set starting values

    a_chain = [param_start[0]]
    f_chain = [param_start[1]]
    fdot_chain = [param_start[2]]

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

    # Initial signal
    
    signal_init_t = waveform(a_chain[0],f_chain[0],fdot_chain[0],t)   # Initial time domain signal
    signal_init_f = FFT(signal_init_t)  # Intial frequency domain signal

                                            
    # Initial value for log posterior
    lp = []
    lp.append(lpost(data_f, signal_init_f, variance_noise_f, a_chain[0], f_chain[0],fdot_chain[0]))  # Append first value of log posterior
    
    lp_store = lp[0]  # Create log posterior storage to be overwritten
                 
    #####                                                  
    # Run MCMC
    #####
    accept_reject_count = [1]

    for i in range(1, Ntotal):

        if i % printerval == 0: # Print accept/reject ratio.
            print("i = ", i, "accept_reject =",sum(accept_reject_count)/len(accept_reject_count))

        # if i % save_interval == 0:
        #     true_vals_for_plot = [true_vals[0],np.log10(true_vals[1]), np.log10(true_vals[2])]

            # os.chdir('/Users/oburke/Documents/LISA_Science/Tutorials/Bayesian_Statistics_Tutorial/MCMC/plots/still_images_single_post_A')
            # fig = plt.hist(a_chain,bins = 30,density = True)
            # plt.axvline(true_vals[0],color = 'red', label = 'True value')            
            # plt.axvline(a_prop,color = 'green', label = 'Proposed value')
            # plt.legend(loc = 'upper right',fontsize = 12)
            # plt.xlabel(r'Parameter $\theta$',fontsize = 12)
            # plt.xlim([0.9777887399340167,1.021990826305361])
            # plt.grid()
            # plt.xticks([])
            # plt.yticks([])

            # plt.ylabel(r'Posterior',fontsize = 12)
            # plt.title("Metropolis Algorithm",fontsize = 15)
            # plt.tight_layout()
            # plt.savefig("a_chain_hist" + str(j) + ".png")
            # plt.clf()
            # j+=1

            # os.chdir('/Users/oburke/Documents/LISA_Science/Tutorials/Bayesian_Statistics_Tutorial/MCMC/plots/still_images_single_trace')
            # plt.plot(a_chain, color = 'purple')
            # plt.plot(true_vals[0]*np.ones(Ntotal),c = 'red',label = 'True value')
            # plt.xlabel('Iteration',fontsize = 10)
            # plt.ylabel(r"Parameter: $\theta$", fontsize = 12)
            # plt.title("Trace plots")
            # plt.yticks([])
            # plt.tight_layout()
            # plt.legend(loc = "lower right", fontsize = 15)
            # plt.savefig("trace_plot_single" + str(j) +".png")
            # plt.clf()
            # j+=1

        norm = np.sqrt(sum((abs(signal_prop_f)**2) / variance_noise_f))
        matched_filter = (1/norm) * np.real(sum(np.conjugate(signal_prop_f)*data_f / variance_noise_f))
        matched_filter_vec.append(matched_filter)

        if i % save_interval == 0:
            burnin = 6000
            # if i <= burnin:
            #     os.chdir('/Users/oburke/Documents/LISA_Science/Tutorials/Bayesian_Statistics_Tutorial/MCMC/plots/still_images_waveform_plots')
                
            #     plt.plot(t_hour, noise_t_plot, alpha = 0.7, c = 'grey', label = 'Noise')
            #     plt.plot(t_hour,waveform(true_vals[0],true_vals[1],true_vals[2],t), alpha = 0.8, c = 'red', label = 'True waveform')
            #     plt.plot(t_hour,waveform(a_prop,f_prop,fdot_prop,t), linestyle='dashed', alpha = 1, c = 'purple', label = 'Proposed waveform')
            #     plt.legend(fontsize = 12, loc = 'upper left')
            #     plt.xlabel(r'Time [hours]', fontsize = 15)
            #     plt.ylabel(r'Strain',fontsize = 15)
            #     plt.title("Matching waveforms", fontsize = 15)
            #     plt.xlim([119.5,120])
            #     plt.savefig("waveform_plot_" + str(j1) +".png")
            #     plt.clf()
            #     plt.close()
            #     j1+=1

            # if i <= burnin:
            #     os.chdir('/Users/oburke/Documents/LISA_Science/Tutorials/Bayesian_Statistics_Tutorial/MCMC/plots/still_images_matched_filter')
            #     plt.plot(matched_filter_vec, label = 'Matched filter SNR')
            #     plt.axhline(y = opt_SNR, c = 'red', label = 'Optimal SNR')
            #     plt.xlim([0,burnin])
            #     plt.ylim([50,200])
            #     plt.ylabel('Strength',fontsize = 15)
            #     plt.xlabel(r'Iterations', fontsize = 15)
            #     plt.title("Matched Filtering SNR",fontsize = 15)
            #     plt.legend(loc = 'lower right', fontsize = 15)
            #     plt.savefig("matched_filter_plot_" + str(j2) + ".png")
            #     plt.clf()
            #     plt.close()

            #     j2+=1

            # if i <= burnin: # before burnin
            #     os.chdir('/Users/oburke/Documents/LISA_Science/Tutorials/Bayesian_Statistics_Tutorial/MCMC/plots/still_images_trace_plot_before_burnin')

            #     samples = [np.log10(a_chain), np.log10(f_chain), np.log10(fdot_chain)]
            #     true_vals_for_plot = [np.log10(true_vals[0]),np.log10(true_vals[1]), np.log10(true_vals[2])]
            #     param_label = [r'$\log_{10}(a)$',r'$\log_{10}(f)$',r'$\log_{10}(\dot{f})$']
            #     color = ['green','black','purple']
            #     fig,ax = plt.subplots(3,1)
            #     for k in range(0,3):
            #         ax[k].plot(samples[k], color = color[k])
            #         ax[k].plot(true_vals_for_plot[k]*np.ones(Ntotal),c = 'red',label = 'True value')
            #         ax[k].set_xlabel('Iteration',fontsize = 10)
            #         ax[k].set_ylabel(param_label[k], fontsize = 10)
            #         ax[k].set_xlim([0,burnin])
            #     ax[0].set_title("Trace plots")
                
            #     plt.tight_layout()
            #     plt.savefig("trace_plot_" + str(j3) +".png")
            #     plt.clf()
            #     plt.close()
            #     j3+=1

            if i > burnin: # after burnin
                os.chdir('/Users/oburke/Documents/LISA_Science/Tutorials/Bayesian_Statistics_Tutorial/MCMC/plots/still_images_trace_plot_after_burnin')

                samples = [np.log10(a_chain), np.log10(f_chain), np.log10(fdot_chain)]
                true_vals_for_plot = [np.log10(true_vals[0]),np.log10(true_vals[1]), np.log10(true_vals[2])]
                param_label = [r'$\log_{10}(a)$',r'$\log_{10}(f)$',r'$\log_{10}(\dot{f})$']
                color = ['green','black','purple']
                fig,ax = plt.subplots(3,1)
                max_min_vec = [[np.log10(4.9294905743897544e-21),np.log10(5.099693103613413e-21)], [np.log10(0.000999963664760359), np.log10(0.0010000495624500195)],
                        [np.log10(9.999679979840393e-09),np.log10(1.0000217263850467e-08)]]
                for k in range(0,3):
                    ax[k].plot(samples[k], color = color[k])
                    ax[k].plot(true_vals_for_plot[k]*np.ones(Ntotal),c = 'red',label = 'True value')
                    ax[k].set_xlabel('Iteration',fontsize = 10)
                    ax[k].set_ylabel(param_label[k], fontsize = 10)
                    ax[k].set_ylim([max_min_vec[k][0], max_min_vec[k][1]])
                    ax[k].set_xlim([burnin,Ntotal])
                ax[0].set_title("Trace plots")

                plt.tight_layout()
                plt.savefig("trace_plot_" + str(j4) +".png")
                plt.clf()
                plt.close()
                j4+=1

        # if i % int(save_interval) ==0 :
            if i > burnin: 
                true_vals_for_plot = [np.log10(true_vals[0]),np.log10(true_vals[1]), np.log10(true_vals[2])]
                os.chdir('/Users/oburke/Documents/LISA_Science/Tutorials/Bayesian_Statistics_Tutorial/MCMC/plots/still_images_joint_post')
                a_chain_log = np.log10(a_chain[burnin:])
                f_chain_log = np.log10(np.array(f_chain[burnin:]))
                fdot_chain_log = np.log10(np.array(fdot_chain[burnin:]))
                samples = np.column_stack([a_chain_log,f_chain_log,fdot_chain_log])
                figure = corner.corner(samples,bins = 30, color = 'blue',plot_datapoints=False,smooth1d=True,
                                    labels=params, 
                                    label_kwargs = {"fontsize":12},set_xlabel = {'fontsize': 20},
                                    show_titles=True, title_fmt='.7f',title_kwargs={"fontsize": 9},smooth = True)

                axes = np.array(figure.axes).reshape((N_param, N_param))

                proposed_vals = [np.log10(a_prop),np.log10(f_prop),np.log10(fdot_prop)]

                for k in range(N_param):
                    ax = axes[k, k]
                    ax.axvline(true_vals_for_plot[k], color="r")
                    ax.axvline(proposed_vals[k], color = "g")
                    
                for yi in range(N_param):
                    for xi in range(yi):
                        ax = axes[yi, xi]
                        ax.axhline(true_vals_for_plot[yi], color="r")
                        ax.axvline(true_vals_for_plot[xi],color= "r")
                        
                        ax.axhline(proposed_vals[yi], color="g")
                        ax.axvline(proposed_vals[xi],color= "g")
                        
                        ax.plot(true_vals_for_plot[xi], true_vals_for_plot[yi], "sr")
                        ax.plot(proposed_vals[xi], proposed_vals[yi],"sg")
                        
                for ax in figure.get_axes():
                    ax.tick_params(axis='both', labelsize=8)
                plt.tight_layout()
                plt.savefig("joint_post" + str(j5) + ".png")
                j5+=1
                plt.clf() 
                plt.close()

        

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