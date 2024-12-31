import os
import librosa
import torch
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Paths to clean and noisy datasets
clean_signals_dir = "data/clean"
noisy_signals_dir = "data/mixed"

# Function to load audio files and compute their Mel spectrogram


# Get lists of files from clean and noisy directories
clean_files = [os.path.join(clean_signals_dir, f) for f in os.listdir(clean_signals_dir) if f.endswith('.wav')]
noisy_files = [os.path.join(noisy_signals_dir, f) for f in os.listdir(noisy_signals_dir) if f.endswith('.wav')]

# Get Mel spectrograms for the first two files in each dataset
clean_spectrograms = [get_mel_spectrogram(f) for f in clean_files[:2]]
noisy_spectrograms = [get_mel_spectrogram(f) for f in noisy_files[:2]]

# Plot the first Mel spectrogram of clean data
plt.figure(figsize=(10, 6))
librosa.display.specshow(clean_spectrograms[0], x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Clean Signal Mel Spectrogram (Example 1)')
plt.show()

# Plot the second Mel spectrogram of clean data
plt.figure(figsize=(10, 6))
librosa.display.specshow(clean_spectrograms[1], x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Clean Signal Mel Spectrogram (Example 2)')
plt.show()

# Plot the first Mel spectrogram of noisy data
plt.figure(figsize=(10, 6))
librosa.display.specshow(noisy_spectrograms[0], x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Noisy Signal Mel Spectrogram (Example 1)')
plt.show()

# Plot the second Mel spectrogram of noisy data
plt.figure(figsize=(10, 6))
librosa.display.specshow(noisy_spectrograms[1], x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Noisy Signal Mel Spectrogram (Example 2)')
plt.show()

# Optionally, convert the spectrograms to PyTorch tensors if needed
clean_spectrograms_tensor = [torch.tensor(spectrogram) for spectrogram in clean_spectrograms]
noisy_spectrograms_tensor = [torch.tensor(spectrogram) for spectrogram in noisy_spectrograms]

# Print the shape of the Mel spectrograms for verification
print("Shape of clean spectrograms:", [s.shape for s in clean_spectrograms])
print("Shape of noisy spectrograms:", [s.shape for s in noisy_spectrograms])
