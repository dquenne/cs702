import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from midiutil import MIDIFile

# load and scale .wav uncompressed audio file
#   filename:     name of file (*.wav)
def load_wav(filename):
  fs, data = wavfile.read(filename)
  return np.array(data) / (2**16.)

# write midi file from 2D array of midi notes
#   note_grid:    2D array of midi note volumes
#   filename:     output filename (*.mid)
#   spacing:      spacing between frames of midi notes (in seconds)
#   duration:     duration of every note
# tempo is generically specified as 60 bpm = 1 beat per second
# a beat-tracking component would be necessary for further specification
def write_midi(note_grid, filename, spacing, duration=1):
  midifile = MIDIFile(1)
  midifile.addTempo(0, 0, 60)
  for i, notes in enumerate(note_grid):
    time = i*spacing
    for k, level in enumerate(notes):
      if level > 0.1 and (i == 0 or (i > 0 and prev_notes[k] < 0.1)):
        midifile.addNote(0, 0, k, float(time), duration, 100)
    prev_notes = notes

  with open(filename, 'wb') as outfile:
    midifile.writeFile(outfile)

# common conversions

def fourier_to_hz(freq, fw, br):
  return freq*br/fw

def hz_to_fourier(freq, fw, br):
  return int(np.round(freq*fw/br))

def hz_to_midi(freq):
  return 1+int(np.ceil(np.log2(freq/440)*12) + 69)

def fourier_to_midi(freq, fw, br):
  return hz_to_midi(fourier_to_hz(freq, fw, br))

# for period-based algorithms, weight to avoid multiples of true period
def weighted_hypothesis(data):
  best = 0
  best_weight = 0
  for j in range(10, len(data)):
    weight = data[j]/np.log(j+10)
    if weight > best_weight:
      best = j
      best_weight = weight
  return best, data[best]

# common display resources

freqlabels_log = [50, 100, 200, 400, 800, 1600, 3200, 6000]

def freqticks_log(fw, br):
  ticks = []
  for f in freqlabels_log:
    ticks.append(hz_to_fourier(f, fw, br))
  return ticks

freqlabels = [400, 800, 1600, 2400, 3200, 4800, 6000]

def freqticks(fw, br, upper_bound = 0):
  ticks = []
  for f in freqlabels:
    k = hz_to_fourier(f, fw, br)
    if upper_bound > 0 and k > upper_bound:
      break
    ticks.append(k)
  return ticks

seclabels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

def periodticks(br=44100):
  ticks = []
  for f in freqlabels_log:
    ticks.append(br/f)
  return ticks

def secticks(fw, br):
  ticks = []
  for t in seclabels:
    ticks.append(fourier_to_hz(t, fw, br))
  return ticks

notelabels = ["C2", "G2", "C3", "G3", "C4", "G4", "C5", "G5", "C6", "G6"]
noteticks  = [36, 43, 48, 55, 60, 67, 72, 79, 84, 91]

def plot_spectrogram(spectra, fw, spacing, br, upper_bound=0):
  H = np.array(spectra)
  if upper_bound > 0:
    plt.imshow(H.T[:upper_bound], origin='lower',aspect='auto',interpolation='nearest')
    plt.yscale('log')
    plt.yticks(freqticks(fw, br, upper_bound), freqlabels)
  else:
    plt.imshow(H.T, origin='lower',aspect='auto',interpolation='nearest')
    plt.yscale('log')
    plt.yticks(freqticks(fw, br), freqlabels)
  plt.ylabel('frequency (Hz)')

  plt.xlabel('time (seconds)')
  sec_length = int(len(spectra)*spacing/br)
  plt.xticks(secticks(spacing, br)[:sec_length], seclabels[:sec_length])
  plt.show()

def plot_correlogram(data, fw, spacing, br, upper_bound=0):
  H = np.array(data)
  plt.imshow(H.T, origin='lower',aspect='auto',interpolation='nearest')
  plt.ylabel('period (samples)')
  plt.xlabel('time (seconds)')
  sec_length = int(len(data)*spacing/br)
  plt.xticks(secticks(spacing, br)[:sec_length], seclabels[:sec_length])
  plt.show()

def plot_pitches(pitches, spacing, br, log=True, fw=1):
  plt.plot(pitches)
  plt.ylabel('frequency (Hz)')
  if log:
    plt.yscale('log')
  plt.yticks(freqlabels_log, freqlabels_log)
  plt.xlabel('time (seconds)')
  sec_length = int(len(pitches)*spacing/br)
  plt.xticks(secticks(spacing, br)[:sec_length], seclabels[:sec_length])
  plt.show()

def plot_peaks(peaks, fw, br, barwidth=3.0):
  plt.bar(peaks.keys(), peaks.values(), barwidth, color='black')
  plt.xticks(freqticks(fw, br)[:-2], freqlabels[:-2])
  plt.xlabel('frequency (Hz)')
  plt.ylabel('amplitude')
  plt.show()

def plot_midi(notes, spacing, br):
  plt.title("piano roll")
  plt.imshow(np.array(notes).T, origin='lower', aspect='auto', interpolation='nearest')
  plt.xlabel('time (seconds)')
  sec_length = int(len(notes)*spacing/br)
  plt.xticks(secticks(spacing, br)[:sec_length], seclabels[:sec_length])
  plt.show()
 
