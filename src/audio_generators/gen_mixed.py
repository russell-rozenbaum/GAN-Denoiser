import os
import random
from scipy.io.wavfile import read, write
import numpy as np
from audio_mixer import mix_audio 

# Paths
clean_signals_dir = "data/pure"
noise_signals_dir = "data/noise"
mixed_signals_dir = "data/mixed"

# Ensure the mixed directory exists
os.makedirs(mixed_signals_dir, exist_ok=True)

# Get list of clean and noise files
clean_files = [f for f in os.listdir(clean_signals_dir) if f.endswith(".wav")]
noise_files = [f for f in os.listdir(noise_signals_dir) if f.endswith(".wav")]

# Mixing process
for clean_file in clean_files:
    # Read clean signal
    clean_path = os.path.join(clean_signals_dir, clean_file)
    sample_rate_clean, clean_signal = read(clean_path)

    # Randomly select a noise file
    noise_file = random.choice(noise_files)
    noise_path = os.path.join(noise_signals_dir, noise_file)
    sample_rate_noise, noise_signal = read(noise_path)

    # Check if sample rates match
    if sample_rate_clean != sample_rate_noise:
        raise ValueError(f"Sample rates do not match for {clean_file} and {noise_file}")

    # Mix the audio
    mixed_signal = mix_audio(clean_signal, noise_signal, snr=0.0)

    # Save the mixed signal
    mixed_file_name = f"mixed_{os.path.splitext(clean_file)[0]}_{os.path.splitext(noise_file)[0]}.wav"
    mixed_path = os.path.join(mixed_signals_dir, mixed_file_name)
    write(mixed_path, sample_rate_clean, mixed_signal.astype(np.int16))

print(f"Mixed audio files saved in '{mixed_signals_dir}'.")
