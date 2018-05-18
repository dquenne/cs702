#!/usr/bin/env python

# compute and display a frame-wise pitch estimation of an audio recording
# using the cepstrum

import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from tqdm import tqdm
import numpy as np
import sys
import util

# get input filename
if len(sys.argv) == 2:
  input_file = sys.argv[1]
else:
  print("usage: ./cepstrum.py input.wav")
  sys.exit()

# basic i/o constants
frame_width = 2048
hamming = np.hamming(frame_width)
spacing = 1024
bitrate = 44100

# storage
data = util.load_wav(input_file)
spectrogram = []
cepstrum = []
best_frequencies = []

for i in tqdm(range(0, int((len(data)-frame_width)/spacing))):
  frame = data[i*spacing:i*spacing+frame_width]
  frame = frame*hamming
  complex_fourier = fft(frame)
  fft_len = int(np.floor(len(complex_fourier)/2))
  power_sp = np.log(abs(complex_fourier))

  spectrogram.append(power_sp[:(fft_len-1)])
  cepst = abs(ifft(power_sp))[:fft_len//2]/frame_width
  cepstrum.append(cepst)
  cepst[0:8] = np.zeros(8) # first several indices give strong false positives
  maxperiod = np.argmax(cepst[30:]) + 30
  best_frequencies.append(bitrate/maxperiod)

plt.title("cepstrum pitches")
util.plot_pitches(best_frequencies, spacing, bitrate)

plt.title("fourier power spectrogram")
util.plot_spectrogram(spectrogram, frame_width, spacing, bitrate)

plt.title("cepstrum spectrogram")
util.plot_correlogram(cepstrum, frame_width, spacing, bitrate)
