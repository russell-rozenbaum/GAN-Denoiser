import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

# Parameters
sample_rate = 44100  # Samples per second
low_freq = 400
high_freq = 4000
duration = 2  # Duration in seconds
num_files = 4000  # Number of noise samples
amplitude = 0.5  # Amplitude of the wave (max = 1.0)
output_dir = "pure"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

for i in range(num_files):
    frequency = np.random.uniform(low=low_freq, high=high_freq)
    # Generate the time points
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Generate the sine wave
    waveform = amplitude * np.sin(2 * np.pi * frequency * t)
    print(waveform)

    # Convert waveform to 16-bit PCM format
    waveform_int16 = (waveform * 32767).astype(np.int16)

    # Save to a .wav file
    output_file = os.path.join(output_dir, f"sine_wave_{i+1:04d}.wav")
    write(output_file, sample_rate, waveform_int16)

print(f"Generated {num_files} noise signals in '{output_dir}' directory.")

# Plot the waveform (first 1000 samples)
'''
plt.plot(t[:1000], waveform[:1000])
plt.title("Sine Wave (440 Hz)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()
'''