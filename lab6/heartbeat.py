import scipy as sp
from scipy.signal import butter, lfilter, filtfilt
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftfreq, ifft
import heartpy as hp

def myplot(x, y, xlabel = 'x', ylabel = 'y'):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

#read file and calculate data
sample_rate, y = wavfile.read('LAB5_500HzFHR.wav')

T = 1/sample_rate
N = len(y)
t=np.linspace(0, N*T, N)

#plot in time domain
myplot(t, y, 't', 'amplitude')

#plot in freq domain
y_fft = fft(y)
x_fft = fftfreq(N, T)
myplot(x_fft, y_fft.real, 'f', 'amplitude')

#remove symetry along y-axis and plot
y_fft = y_fft[0:N//2]
x_fft = x_fft[0:N//2]
myplot(x_fft, y_fft.real, 'f', 'amplitude')
#remove symetry along x-axis and plot
myplot(x_fft, 2.0/N*np.abs(y_fft[0:N//2]), 'f', 'amplitude')

#region filters - comment butterworth or manual, depending which one you want to use (if not manual will be used)
f1 = 100
f2 = 200

#filter with 9th Butterworth and plot results
y_filtered = butter_bandpass_filter(y, f1, f2, sample_rate, 9)
#myplot(x_fft, 2.0/N*np.abs(y_filtered[0:N//2]), 'f', 'amplitude')
myplot(t, y_filtered, 't', 'amplitude')

# plt.figure()
# plt.plot(t, y, 'r', label='original')
# plt.plot(t, y_filtered, 'b', label='filtered')
# plt.legend()
# plt.xlabel('t')
# plt.ylabel('amplitude')
# plt.show()

#filter manually and plot results
deltaf = x_fft[-1] / (N//2)
f1i = int(f1//deltaf)
f2i = int(f2//deltaf)
y_filtered = y_fft
y_filtered[:f1i] = 0
y_filtered[f2i:] = 0

myplot(x_fft, 2.0/N*np.abs(y_fft[0:N//2]), 'f', 'amplitude')
 
y_filtered = ifft(y_filtered).real
myplot(t[:len(t)//2], y_filtered, 't', 'amplitude')
#endregion

#convert to int16 from float
y_int16 = sp.int16(y_filtered/sp.absolute(y_filtered).max() * 32767)

#plot
wavfile.write('heartbeatFiltered.wav', sample_rate, y_int16)


#region heartpy

#process unfiltered data with heartpy
working_data, measures = hp.process(y, sample_rate)
print("BPM for unfiltered:" + str(measures['bpm']))
hp.plotter(working_data, measures)
input("Press Enter to continue...")                     #to make sure plot won't close itself

#process filtered data with heartpy
working_data, measures = hp.process(y_filtered, sample_rate)
print("BPM for filtered:" + str(measures['bpm']))
hp.plotter(working_data, measures)
input("Press Enter to continue...")                     #to make sure plot won't close itself

#endregion
