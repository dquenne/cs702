#!/usr/bin/env python

# compute multi-pitch estimation of an audio recording and output as MIDI file

# based on an iterative version of Pertusa and Inesta's 2008 paper

import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from tqdm import tqdm
import numpy as np
import itertools
import sys
import util # commonly used functions for pitch detection

# get i/o filenames
if len(sys.argv) == 3:
  input_file = sys.argv[1]
  output_file = sys.argv[2]
else:
  print("usage: ./pertusa-inesta.py input.wav out.mid")
  sys.exit()

# basic i/o constants
frame_width = fw = 8192
hann = np.hanning(frame_width)
spacing = frame_width//4
bitrate = br = 44100

# algorithm constants
f_0 = 50
f_1 = 2000
f_r = 10
power_thresh = 5
big_f = 10
gaussian = [0.21, 0.58, 0.21]
gamma = 0.1

use_zeropad = True

# storage
data = util.load_wav(input_file)
f0_weights = []
midi_result = []
all_notes = []
if use_zeropad:
  fft_len = fw*4
  zeropad = np.zeros(fw*3)
else:
  fft_len = fw
  zeropad = np.array([])

# convert relevant frequency range to FFT indices
k0 = int(np.floor(util.hz_to_fourier(f_0, fw*4, br)))
k1 = int(np.ceil(util.hz_to_fourier(f_1, fw*4, br)))

# get spectral peaks and frequency hypotheses from a power spectrum
def get_spectral_peaks_and_hypotheses(spectrum):
  peaks = {}
  hypotheses = []
  prev = spectrum[0]
  higher = spectrum[1]
  for k in range(1, len(spectrum)-1):
    power = higher
    higher = spectrum[k+1]
    if (power > power_thresh):
      if power > prev and power > higher:
        peaks[k] = power
        if k > k0 and k < k1: # frequency range for potential F0s
          hypotheses.append(k)
    prev = power
  return peaks, hypotheses

# iterate through frames
for i in tqdm(range(0, int((len(data)-frame_width)/spacing))):

  # spectrum generation and preprocessing

  frame = data[i*spacing:i*spacing+frame_width]
  window = frame * hann
  raw_fft = fft(np.concatenate((window, zeropad)))
  spectrum_len = int(np.floor(len(raw_fft)/2))
  power_sp = abs(raw_fft)[:spectrum_len]

  peaks, hypotheses = get_spectral_peaks_and_hypotheses(power_sp)

  # compute initial loudnesses for F0 candidates

  total_powers = np.zeros(k1)
  patterns = {}
  harmonic_owners = {} # hashmap of harmonic ownership
  loudnesses = {}
  for f0 in hypotheses:
    patterns[f0] = [(f0, power_sp[f0])]
    total_powers[f0] = power_sp[f0]
    if f0 in harmonic_owners:
      harmonic_owners[f0].append(f0)
    else:
      harmonic_owners[f0] = [f0]

    for harmonic in range(2, 20):
      best = 0
      best_freq = 0
      if harmonic*f0 > len(power_sp):
        break
      for inharmonicity in range(-f_r, min(f_r, spectrum_len)):
        h_f = harmonic*f0 + inharmonicity
        if h_f in peaks:
          h_power = power_sp[h_f]
          if h_power > best:
            best = h_power
            best_freq = h_f
      if best_freq == 0:
        best_freq = harmonic*f0
        peaks[harmonic*f0] = 0.0
      if best_freq in harmonic_owners:
        harmonic_owners[best_freq].append(f0)
      else:
        harmonic_owners[best_freq] = [f0]

      total_powers[f0] += best
      patterns[f0].append((best_freq, best))
    loudnesses[f0] = total_powers[f0]

  kept_notes = []
  for dummy in range(0, big_f):
    best = np.argmax(total_powers)
    if total_powers[best] <= 90.0:
      break
    total_powers[best] = 0.0
    big_l = 0
    smallest_l = 100000
    p = patterns[best]
    peaks[p[0][0]] = 0.0
    for h in range(1, len(p)-1):
      h_f = p[h][0]
      if len(harmonic_owners[h_f]) > 1: # shared harmonic
        h_fl = p[h-1][0]
        h_fr = p[h+1][0]
        interpolated = (p[h-1][1] + peaks[h_fr])/2
        if interpolated < peaks[h_f]:
          p[h] = (h_f, interpolated)
          peaks[h_f] = peaks[h_f] - interpolated
          for owner in harmonic_owners[h_f]:
            total_powers[owner] -= interpolated
        else:
          p[h] = (h_f, peaks[h_f])
          peaks[h_f] = 0.0
          for owner in harmonic_owners[h_f]:
            total_powers[owner] -= peaks[h_f]
      else:
        if h_f < len(peaks):
          p[h] = (h_f, peaks[h_f])
          peaks[h_f] = 0 # this is just for visualizing spectral subtraction
    powers = np.array(p).T[1]
    loudness = np.sum(powers)
    if loudness > big_l:
      big_l = loudness
    if loudness < smallest_l:
      smallest_l = loudness
    if smallest_l > gamma*big_l:
      kept_notes.append(best)

  result = np.zeros(k1)
  midi_notes = []
  for freq in kept_notes:
    hzfreq = util.fourier_to_hz(freq, fft_len, br)
    note = util.hz_to_midi(hzfreq)
    midi_notes.append(note)
    result[freq] = loudnesses[freq]
  f0_weights.append(result)

  all_notes.append(midi_notes)

# post processing (note pruning based on duration)

for i in range(0, len(all_notes)):
  midi_array = np.zeros(140)
  for note in all_notes[i]:
    if (i >= 2 and i+2 < len(all_notes)):
      if note in all_notes[i-1] and note in all_notes[i-2] or \
        note in all_notes[i-1] and note in all_notes[i+1] or \
        note in all_notes[i+1] and note in all_notes[i+2]:
        midi_array[note] = 1.0
      elif note+1 in all_notes[i-1] and note+1 in all_notes[i+1]:
        midi_array[note+1] = 1.0
      elif note-1 in all_notes[i-1] and note-1 in all_notes[i+1]:
        midi_array[note-1] = 1.0
    elif (i < 2 or i+2 >= len(all_notes)-1):
      midi_array[note] = 1.0

  midi_result.append(midi_array)

print('writing midi file')
util.write_midi(midi_result, output_file, spacing/br, 4)
