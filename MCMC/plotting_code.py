import matplotlib.pyplot as plt
import numpy as np
import os

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