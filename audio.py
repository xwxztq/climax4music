import scipy.io.wavfile as wavfile
import scipy
from scipy import signal
import scipy.fftpack
import numpy as np
from matplotlib import pyplot as plt
import stft


sample_rate, sample = wavfile.read("static/小提琴 - 梁祝.wav")

channel = len(sample.shape)
if channel == 2:
    sample = sample.sum(axis=1) / 2
N = sample.shape[0]
secs = N / float(sample_rate)
Ts = 1.0/sample_rate

print("Frequency sampling", sample_rate)
print("Channels", channel)
print("Complete Samplings N", N)
print("secs", secs)
print("Timestep between samples Ts", Ts)

t = scipy.arange(0, secs, Ts)
FFT = abs(scipy.fft(sample))
# one side FFT range
FFT_side = FFT[range(N//2)]
freqs = scipy.fftpack.fftfreq(sample.size, t[1]-t[0])
fft_freqs = np.array(freqs)
# one side frequency range
freqs_side = freqs[range(N//2)]
fft_freqs_side = np.array(freqs_side)

# render part
plt.subplot(311)
# plotting the sample
p1 = plt.plot(t, sample, "g")
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.subplot(312)
# plotting the complete fft spectrum
p2 = plt.plot(freqs, FFT, "r")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Count dbl-sided')
plt.subplot(313)
# plotting the positive fft spectrum
p3 = plt.plot(freqs_side, abs(FFT_side), "b")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Count single-sided')
plt.show()

# plotting the stft spectrum
# stft part
specgram = stft.spectrogram(sample)
output = stft.ispectrogram(specgram)
wavfile.write('output.wav', sample_rate, output)

"""
frequencies, times, specgram_ = signal.spectrogram(sample, sample_rate)
plt.pcolormesh(times, frequencies, specgram_)
plt.imshow(specgram_, origin="lower", aspect="auto", cmap="hot")
plt.colorbar()
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time [sec]')
plt.show()
"""

plt.specgram(sample, NFFT=1024, Fs=sample_rate, cmap="hot")
plt.show()