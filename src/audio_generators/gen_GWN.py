import numpy as np
import os
from scipy.io.wavfile import write

# Parameters
sample_rate = 44100  # Samples per second
duration = 2  # Duration in seconds
num_files = 400  # Number of noise samples
amplitude = 0.5  # Amplitude of the noise (max = 1.0)
output_dir = "noise"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

for i in range(num_files):
    # Generate Gaussian white noise
    noise = np.random.normal(0, amplitude, int(sample_rate * duration))

    # Convert noise to 16-bit PCM format
    noise_int16 = (noise * 32767).astype(np.int16)

    # Save to a .wav file
    output_file = os.path.join(output_dir, f"noise_{i+1:04d}.wav")
    write(output_file, sample_rate, noise_int16)

print(f"Generated {num_files} noise signals in '{output_dir}' directory.")
