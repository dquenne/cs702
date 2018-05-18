#!/usr/bin/env python

# compute and display single-pitch estimation of an audio recording based on 
# harmonicity model

import matplotlib.pyplot as plt
from scipy.fftpack import fft
from tqdm import tqdm
import numpy as np
import sys
import util # commonly used functions for pitch detection

# basic i/o constants
frame_width = 8192
hann = np.hanning(frame_width)
spacing = 1024
bitrate = 44100

f_0 = 50
f_1 = 2000
f_r = 10
power_thresh = 10

if len(sys.argv) == 2:
  input_file = sys.argv[1]
else:
  print("usage: ./harmonicity.py input.wav")
  sys.exit()

data = util.load_wav(input_file)
all_weights = []
fft_len = frame_width*4
zeropad = np.zeros(frame_width*3)
best_frequencies = []

k0 = int(np.floor(util.hz_to_fourier(f_0, frame_width*4, bitrate)))
k1 = int(np.ceil(util.hz_to_fourier(f_1, frame_width*4, bitrate)))
# iterate through frames
for i in tqdm(range(0, int((len(data)-frame_width)/spacing))):

  # spectrum generation and preprocessing

  frame = data[i*spacing:i*spacing+frame_width]
  window = frame * hann
  raw_fft = fft(np.concatenate((window, zeropad)))
  spectrum_len = int(np.floor(len(raw_fft)/2))
  power_sp = abs(raw_fft)[:spectrum_len]

  # actual F0 estimation part

  hypotheses = []
  peaks = {}

  prev = power_sp[0]
  higher = power_sp[1]
  for k in range(1, len(power_sp)-1):
    power = higher
    higher = power_sp[k+1]
    if (power > power_thresh):
      if power > prev and power > higher:
        peaks[k] = power
        if k > k0 and k < k1:
          hypotheses.append(k)
    prev = power

  total_powers = np.zeros(k1)
  for f0 in hypotheses:
    total_powers[f0] = power_sp[f0]
    for harmonic in range(2, 20):
      best_power = 0
      best_freq = 0
      if harmonic*f0 + f_r > len(power_sp):
        break
      for inharmonicity in range(-f_r, f_r):
        h_f = harmonic*f0 + inharmonicity
        if h_f in peaks:
          h_power = power_sp[h_f]
          if h_power > best_power:
            best_power = h_power
            best_freq = h_f

      total_powers[f0] += best_power

  all_weights.append(total_powers)
  best_frequencies.append(np.argmax(total_powers))

plt.title("harmonicity-based pitches")
util.plot_pitches(best_frequencies, spacing, bitrate)

# display spectra:

plt.title("harmonicity-based loudness weights")
util.plot_spectrogram(all_weights, fft_len, spacing, bitrate)
