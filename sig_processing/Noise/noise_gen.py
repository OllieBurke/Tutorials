import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import tukey
import os

os.chdir('/Users/oburke/Documents/LISA_Science/Tutorials/Bayesian_Statistics_Tutorial/sig_processing/Noise/Plots')
delta_t = 10
t = np.arange(0,7*24*60*60,delta_t)
tmax = t[-1]
N_t = len(t)
PSD = 1e-22 
np.random.seed(1234)

noise_t = np.random.normal(0,PSD,N_t)

# plt.plot(t,noise_t)
# plt.xlabel(r'Time [days]',fontsize = 15)
# plt.ylabel(r'Noise Strain',fontsize = 15)
# plt.title(r'Stationary noise',fontsize = 15)
# plt.tight_layout()
# plt.savefig("Stationary_noise.pdf")
# plt.clf()

t_c = 3.5*24*60*60
one_hour = 60*60
start_window = t_c - 11*one_hour           # Define start of gap
end_window = start_window + 8*one_hour          # Define end of gap
lobe_length = 5*one_hour          # Define length of cosine lobes
        # Define length of cosine lobes

window_length = int(np.ceil(((end_window+lobe_length) - 
                             (start_window - lobe_length))/delta_t))  # Construct of length of window 
                                                                      # throughout the gap
alpha = 0*2*lobe_length/(delta_t*window_length)      # Construct alpha (windowing parameter)
                                                   # so that we window BEFORE the gap takes place.
    
window = tukey(window_length,alpha)   # Construct window

new_window = []  # Initialise with empty vector
j=0  
for i in range(0,len(t)):   # loop index i through length of t
    if t[i] > (start_window - lobe_length) and (t[i] < end_window + lobe_length):  # if t within gap segment
        new_window.append(np.nan)  # add windowing function to vector of ones.
        j+=1  # incremement 
    else:                   # if t not within the gap segment
        new_window.append(1)  # Just add a onne.
        j=0

noise_t_gap = noise_t*new_window
plt.plot(t/60/60/24,noise_t_gap)
plt.xlabel(r'Time [days]',fontsize = 15)
plt.ylabel(r'Noise Strain',fontsize = 15)
plt.title(r'Gaps in data',fontsize = 15)
plt.tight_layout()
plt.savefig("Gaps_noise.pdf")
plt.clf()

# t_c = 4*24*60*60
# one_hour = 60*60
# start_window = t_c - 24*one_hour           # Define start of gap
# end_window = start_window + 24*one_hour          # Define end of gap
# lobe_length = 10*one_hour          # Define length of cosine lobes
#         # Define length of cosine lobes

# window_length = int(np.ceil(((end_window+lobe_length) - 
#                              (start_window - lobe_length))/delta_t))  # Construct of length of window 
#                                                                       # throughout the gap
# alpha = 2*lobe_length/(delta_t*window_length)      # Construct alpha (windowing parameter)
#                                                    # so that we window BEFORE the gap takes place.
    
# window = tukey(window_length,alpha)   # Construct window

# glitch = []  # Initialise with empty vector
# j=0  
# for i in range(0,len(t)):   # loop index i through length of t
#     if t[i] > (start_window - lobe_length) and (t[i] < end_window + lobe_length):  # if t within gap segment
#         # glitch.append((3 - window[j])*np.sin(0*(t[i]/t[-1]))*np.exp(-t[i]/(10*t[-1])))  # add windowing function to vector of ones.
#         glitch.append((2 - window[j]))  # add windowing function to vector of ones.

#         j+=1  # incremement 
#     else:                   # if t not within the gap segment
#         glitch.append(1)  # Just add a onne.
#         j=0

# plt.plot(t/60/60/24,glitch*noise_t)
# plt.xlabel(r'Time [days]',fontsize = 15)
# plt.ylabel(r'Noise Strain',fontsize = 15)
# plt.title(r'Glitches in data',fontsize = 15)
# plt.tight_layout()
# plt.savefig("Glitches_noise.pdf")
# plt.clf()