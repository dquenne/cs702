#!/usr/bin/env python

# compute and display a frame-wise pitch estimation of an audio recording
# using the autocorrelation function

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import sys
import util

if len(sys.argv) == 2:
  input_file = sys.argv[1]
else:
  print("usage: ./ac.py input.wav")
  sys.exit()

# constants
f_min = 50
f_max = 2000
frame_width = 2048
spacing = 2048
bitrate = 44100

# compute autocorrelation of data
def autocorrelation(data, min_lag, max_lag):
  n = len(data)
  result = list(np.zeros(min_lag))
  for lag in range(min_lag, max_lag):
    sumarray = np.zeros(n+lag)
    sumarray[:n] = data
    sumarray[:n-lag] *= data[lag:]
    sum = np.sum(sumarray[:n-lag])
    result.append(float(sum/(n-lag)))
  return result

data = util.load_wav(input_file)
ac_correlogram = []
best_frequencies = []
hann = np.hanning(frame_width)
for i in tqdm(range(0, int((len(data)-frame_width)/spacing))):
  frame = data[i*spacing:i*spacing+frame_width]
  c = frame*hann
  ach = autocorrelation(c, bitrate // f_max, bitrate // f_min)
  ac_correlogram.append(ach)
  best_lag = np.argmax(ach)
  if best_lag <= 1:
    best_frequencies.append(0)
  else:
    best_frequencies.append(bitrate/best_lag)

plt.title("autocorrelation pitches")
util.plot_pitches(best_frequencies, spacing, bitrate)

plt.title("autocorrelation correllogram")
util.plot_correlogram(ac_correlogram, frame_width, spacing, bitrate)
