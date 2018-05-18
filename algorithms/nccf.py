#!/usr/bin/env python

# compute and display a frame-wise pitch estimation of an audio recording
# using a normalized cross-correlation function

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import sys
import util

if len(sys.argv) == 2:
  input_file = sys.argv[1]
else:
  print("usage: ./nccf.py input.wav")
  sys.exit()

# constants
f_min = 50
f_max = 2000
frame_width = 2048
spacing = 2048
bitrate = bitrate = 44100

# compute cross-correlation and normalized cross-correlation of data
def nccf(data, frame_index, e, min_lag, max_lag):
  n = frame_width
  ccf = list(np.zeros(min_lag))
  nccf = list(np.zeros(min_lag))
  for lag in range(min_lag, max_lag):
    lag_sum = 0
    sumarray = np.zeros(n+lag)
    sumarray[:n] = data[frame_index:frame_index+n]
    sumarray[:n] *= data[frame_index+lag:frame_index+n+lag]
    lag_sum = np.sum(sumarray[:n])
    ccf.append(lag_sum)
    nccf.append(lag_sum/np.sqrt(e[frame_index]*e[frame_index+lag]))
  return ccf, nccf

# storage
data = util.load_wav(input_file)
cc_correlogram = []
nccf_correlogram = []
best_frequencies = []
hann = np.hanning(frame_width)

# pre-calculate normalization factor data
squared_data = []
for i in range(0, len(data)):
  squared_data.append(data[i]**2)
e = [0.0]

for i in range(0, frame_width-1):
  e[0] += squared_data[i]

for i in range(0, len(data)-frame_width):
  e.append(e[i-1]-squared_data[i-1]+squared_data[i+frame_width])

for i in tqdm(range(0, int((len(data)-frame_width*2)/spacing))):
  cc,ncc = nccf(data, i*spacing, e, bitrate // f_max, bitrate // f_min)
  cc_correlogram.append(cc)
  nccf_correlogram.append(ncc)
  best_lag = np.argmax(ncc)
  if best_lag == 0:
    best_frequencies.append(0)
  else:
    best_frequencies.append(bitrate/best_lag)

plt.title("normalized cross-correlation pitches")
util.plot_pitches(best_frequencies, spacing, bitrate)

plt.title("cross-correlation correlogram")
util.plot_correlogram(cc_correlogram, frame_width, spacing, bitrate)

plt.title("normalized cross-correlation correlogram")
util.plot_correlogram(nccf_correlogram, frame_width, spacing, bitrate)
