import numpy as np
import matplotlib.pyplot as plt
import os

def signal_gen_harmonics(harmonics, freq = 1):
    gen_signal = [np.cos(2*np.pi*(n+1)*1*t) for n in range(0,harmonics)]
    return sum(gen_signal)

highest_frequency = 40
delta_t = (1/(2*highest_frequency))    # Sampling interval - change the 14 to one of {3,5,7,11,13} and look at the difference.
f_s = 1/delta_t      # Sampling rate
critical_freq = (1/(0.01*2*delta_t)) # Will be spoken about later.

t = np.arange(0,10,delta_t)    # This creates a vector of times between 0 and 10 evenly spaced by delta_t
signal1 = 1*np.cos(2*np.pi*1*t)
signal2 = 1*np.cos(2*np.pi*2*t)
signal3 = 1*np.cos(2*np.pi*3*t)
signal4 = 1*np.cos(2*np.pi*4*t)
signal5 = 1*np.cos(2*np.pi*5*t) #  Individual signals of different harmonics m = {3,5,7,11,13}.

signal = signal1 + signal2 + signal3 + signal4 + signal5 # Superposition of individual harmonics of signals - sound familiar?
                  
harmonics = 1
signal =  signal_gen_harmonics(harmonics,t)                                                                            # self explanatory

plt.plot(t,signal);plt.xlabel(r'$time [seconds]$',fontsize = 15);plt.ylabel(r'Amplitude',fontsize = 15);plt.title('Plot of sinusoid - 1 harmonics',fontsize = 15);plt.tight_layout();
# plt.savefig("time_domain_sig.pdf")
plt.show()
plt.clf()



n_t = len(signal) # length of signal in the time domain
critical_frequency = 1/(2*delta_t) # Calculate maximum resolvable frequency
freq = np.fft.rfftfreq(n_t,delta_t)
fft_signal = np.fft.rfft(signal)

# Plotting stem plots are slow, so don't run this one every time. You won't see much if you increase 
# "highest frequency" above.
plt.stem(freq, abs(fft_signal)**2);
print('length of the time series in time is %s so power at each frequency is n_t/2 = %s'%(n_t,n_t/2))
plt.xlabel(r'$f$ [Hz]',fontsize = 15);plt.xlim([0,10]);plt.ylabel(r'$|\hat{h}(f)|^2$',fontsize = 15);plt.title('Plot of sinusoid - frequency domain',fontsize = 15);plt.tight_layout();
# plt.savefig("freq_domain_sig.pdf");
plt.show()
plt.clf()


PSD = 0.5   # White noise
sigma_sqr = PSD/(4*delta_t)
np.random.seed(1234)
noise = np.random.normal(0,np.sqrt(sigma_sqr),n_t)
data = signal + noise

plt.plot(t,data);plt.xlabel(r'time [seconds]',fontsize = 15);plt.ylabel(r'Amplitude',fontsize = 15);plt.xlim([0,10]);plt.title('Plot of data stream - time domain',fontsize = 15);plt.tight_layout();
# plt.savefig("data_time_domain.pdf");
plt.show()
plt.clf()

data = signal + noise
plt.plot(t,data,label = 'data');plt.plot(t,signal,'r',label = 'signal');
plt.xlabel(r'$time [seconds]$',fontsize = 15);plt.ylabel(r'$Amplitude$',fontsize = 15);plt.title('Plot of data + signal - time domain',fontsize = 15);
plt.xlim([0,10])
plt.legend()
plt.tight_layout()
plt.show()
# plt.savefig("noise_and_signal_time_domain.pdf");plt.clf()

dft_data = np.fft.rfft(data)
plt.stem(freq,np.abs(dft_data)**2)
plt.xlabel(r'Frequency [Hz]',fontsize = 15)
plt.ylabel(r'$|\hat{d}(f)|^{2}|$',fontsize = 15)
plt.xlim([0,10])
plt.title('Plot of data + signal - frequency domain',fontsize = 15)
plt.tight_layout()
plt.show()
# plt.savefig("data_freq_domain.pdf")
plt.clf()
           

# Signal to noise ratio

SNR2 = (4*delta_t/n_t)*sum(abs(fft_signal)**2 / PSD)
print(np.sqrt(SNR2))  # Optimal matched filter SNR

Fs = 1/delta_t
def compute_snr_timeseries(filter, data, psd, dt):
    # define come constants
    Nt = len(filter)
    data_duration = Nt*dt

    # compute data ffts
    detector_fft = np.fft.rfft(data) * dt
    waveform_fft = np.fft.rfft(filter) * dt
    
    #compute complex conjugate
    finner =  np.conj(waveform_fft) * detector_fft / psd
    #compute the optimal SNR
    osnr_sum = 4 / data_duration * np.sum(np.conj(waveform_fft) * waveform_fft / psd)
    # compute matched filter SNR for optimal waveform
    snr_sum = 4 / data_duration * np.sum(finner) / np.sqrt(osnr_sum)

    # compute the SNR timeseries
    snr_time = 4 / data_duration * 0.5*Nt * np.fft.irfft(finner)/ np.sqrt(osnr_sum)

    # roll the timeseries so template times match up
    return np.roll(snr_time, 0*-int(0.25*data_duration*Fs)), snr_sum, np.sqrt(osnr_sum)

SNR_time, matched_SNR, opt_SNR = compute_snr_timeseries(signal, data, PSD, delta_t)

# print(matched_SNR,np.real(opt_SNR))

fig, ax = plt.subplots(1,2, figsize = (16,8))
ax[0].plot(t, abs(SNR_time))
ax[1].plot(t,signal)
#ax.plot(data_times[100:-100], abs(snr2.data), alpha = 0.5)
ax[0].set_xlabel("Time")
ax[0].set_ylabel("SNR")

ax[1].set_xlabel("Time")
ax[1].set_ylabel("Amplitude")
plt.show()
plt.clf()

quit()
breakpoint()

dft_noise = np.fft.fft(noise)
plt.stem(freq,np.abs(dft_noise)**2)
plt.xlabel(r'frequency bins')
plt.ylabel(r'power $|n(f)|^{2}|$')           
plt.show()
plt.clf()
           
plt.stem(freq,np.abs(dft_data)**2,'r',label = 'data')
plt.stem(freq,np.abs(dft_noise)**2,'blue',label = 'signal')
plt.xlabel(r'frequency bins')
plt.ylabel(r'power $|d(f)|^{2}|$')
plt.show()
plt.clf()