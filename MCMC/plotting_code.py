import matplotlib.pyplot as plt
import numpy as np
import os
from mcmc_fun import waveform
from corner import corner
def plot_PSD(PSD,h_true_f,freq,delta_t):
    """
    Here we plot a plot a comparison of the signal in the frequency domain against the PSD.
    Useful if we wish to determine roughly what the signal-to-noise ratio is by eye.
    """

    os.chdir('/Users/oburke/Documents/LISA_Science/Tutorials/Bayesian_Statistics_Tutorial/MCMC/plots/PSD_plot')
    # os.chdir("DIRECTORY YOU WANT TO SAVE PLOT HERE")
    plt.figure(figsize = (12,8))
    plt.loglog(freq,2*np.sqrt(freq*abs(delta_t*h_true_f)**2),label = r'$2f\cdot |\hat{h}(f)|$')
    plt.loglog(freq,np.sqrt(PSD), label = r'$\sqrt{S_{n}(f)}$')
    plt.legend(fontsize = 20)
    plt.xlabel(r'Frequency [Hz]', fontsize = 20)
    plt.ylabel(r'Magnitude - Fourier domain', fontsize = 20)
    plt.title('Comparison between the PSD and signal', fontsize = 20)
    plt.xlim([1e-4,1e-1])
    plt.ylim([1e-30,1e-14])
    plt.grid()
    plt.xticks(fontsize= 16)
    plt.yticks(fontsize = 16)
    plt.tight_layout()
    plt.savefig("Comparison_signal_PSD.pdf")
    plt.show()
    plt.clf()
    plt.close()

def waveform_plot(j1, t, t_hour, true_vals, a_prop, f_prop, fdot_prop,noise_t_plot, dir):
    waveform_direc = dir + "/waveform_plot"
    # os.chdir('/Users/oburke/Documents/LISA_Science/Tutorials/Bayesian_Statistics_Tutorial/MCMC/plots/still_images_waveform_plots')
    os.chdir(waveform_direc)

    # os.chdir("DIRECTORY YOU WANT TO SAVE PLOT HERE")

    plt.plot(t_hour, noise_t_plot, alpha = 0.7, c = 'grey', label = 'Noise')
    plt.plot(t_hour,waveform(true_vals[0],true_vals[1],true_vals[2],t), alpha = 0.8, c = 'red', label = 'True waveform')
    plt.plot(t_hour,waveform(a_prop,f_prop,fdot_prop,t), linestyle='dashed', alpha = 1, c = 'purple', label = 'Proposed waveform')
    plt.legend(fontsize = 12, loc = 'upper left')
    plt.xlabel(r'Time [hours]', fontsize = 15)
    plt.ylabel(r'Strain',fontsize = 15)
    plt.title("Matching waveforms", fontsize = 15)
    plt.xlim([119.5,120])
    plt.savefig("waveform_plot_" + str(j1) +".png")
    plt.clf()
    plt.close()
    j1+=1
    return j1

def matched_filter_plot(j2,matched_filter_vec, opt_SNR, burnin, dir):
    matched_filter_direc = dir + "/matched_filter"
    # os.chdir('/Users/oburke/Documents/LISA_Science/Tutorials/Bayesian_Statistics_Tutorial/MCMC/plots/still_images_waveform_plots')
    os.chdir(matched_filter_direc)
    # os.chdir("DIRECTORY YOU WANT TO SAVE PLOT HERE")

    plt.plot(matched_filter_vec, label = 'Matched filter SNR')
    plt.axhline(y = opt_SNR, c = 'red', label = 'Optimal SNR')
    plt.xlim([0,burnin])
    plt.ylim([50,200])
    plt.ylabel('Strength',fontsize = 15)
    plt.xlabel(r'Iterations', fontsize = 15)
    plt.title("Matched Filtering SNR",fontsize = 15)
    plt.legend(loc = 'lower right', fontsize = 15)
    plt.savefig("matched_filter_plot_" + str(j2) + ".png")
    plt.clf()
    plt.close()

    j2+=1
    return j2

def trace_plot_before_burnin(j3,a_chain,f_chain,fdot_chain,true_vals,Ntotal,burnin,dir):
    # os.chdir('/Users/oburke/Documents/LISA_Science/Tutorials/Bayesian_Statistics_Tutorial/MCMC/plots/still_images_trace_plot_before_burnin')
    # os.chdir("DIRECTORY YOU WANT TO SAVE PLOT HERE")

    trace_plot_before_burnin_direc = dir + "/trace_plot_before_burnin"
    # os.chdir('/Users/oburke/Documents/LISA_Science/Tutorials/Bayesian_Statistics_Tutorial/MCMC/plots/still_images_waveform_plots')
    os.chdir(trace_plot_before_burnin_direc)
    samples = [np.log10(a_chain), np.log10(f_chain), np.log10(fdot_chain)]
    true_vals_for_plot = [np.log10(true_vals[0]),np.log10(true_vals[1]), np.log10(true_vals[2])]
    param_label = [r'$\log_{10}(a)$',r'$\log_{10}(f)$',r'$\log_{10}(\dot{f})$']
    color = ['green','black','purple']
    fig,ax = plt.subplots(3,1)
    for k in range(0,3):
        ax[k].plot(samples[k], color = color[k])
        ax[k].plot(true_vals_for_plot[k]*np.ones(Ntotal),c = 'red',label = 'True value')
        ax[k].set_xlabel('Iteration',fontsize = 10)
        ax[k].set_ylabel(param_label[k], fontsize = 10)
        ax[k].set_xlim([0,burnin])
    ax[0].set_title("Trace plots")

    plt.tight_layout()
    plt.savefig("trace_plot_" + str(j3) +".png")
    plt.clf()
    plt.close()
    j3+=1
    return j3

def trace_plot_after_burnin(j4,a_chain,f_chain,fdot_chain,true_vals,Ntotal,burnin,dir):
    # os.chdir('/Users/oburke/Documents/LISA_Science/Tutorials/Bayesian_Statistics_Tutorial/MCMC/plots/still_images_trace_plot_after_burnin')
    # os.chdir("DIRECTORY YOU WANT TO SAVE PLOT HERE")

    trace_plot_after_burnin_direc = dir + "/trace_plot_after_burnin"
    # os.chdir('/Users/oburke/Documents/LISA_Science/Tutorials/Bayesian_Statistics_Tutorial/MCMC/plots/still_images_waveform_plots')
    os.chdir(trace_plot_after_burnin_direc)

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
    return j4

def corner_plot_after_burnin(j5,true_vals,a_chain,f_chain,fdot_chain,burnin,params,a_prop,f_prop,fdot_prop,N_param,dir):

    joint_post_direc = dir + "/joint_post"
    # os.chdir('/Users/oburke/Documents/LISA_Science/Tutorials/Bayesian_Statistics_Tutorial/MCMC/plots/still_images_waveform_plots')
    os.chdir(joint_post_direc)

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
    return j5