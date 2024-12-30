import tensorflow as tf
import os
from scipy.io.wavfile import read
import numpy as np

# Paths to data splits
train_clean_dir = "data/splits/train/clean"
train_noisy_dir = "data/splits/train/noisy"
val_clean_dir = "data/splits/val/clean"
val_noisy_dir = "data/splits/val/noisy"
test_clean_dir = "data/splits/test/clean"
test_noisy_dir = "data/splits/test/noisy"

# Parameters
batch_size = 32
sample_rate = 44100  # Assuming all audio is sampled at 44.1 kHz
duration = 2  # Audio clip duration in seconds
clip_length = sample_rate * duration  # Total samples per clip


def load_audio(file_path):
    """Load a .wav file and normalize to [-1, 1]."""
    sample_rate, audio = read(file_path.numpy().decode("utf-8"))
    audio = audio.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
    if len(audio) > clip_length:
        audio = audio[:clip_length]  # Trim if too long
    elif len(audio) < clip_length:
        audio = np.pad(audio, (0, clip_length - len(audio)), mode="constant")  # Pad if too short
    return audio


def tf_load_audio(file_path):
    """Wrapper for TensorFlow to call the load_audio function."""
    audio = tf.py_function(load_audio, [file_path], tf.float32)
    audio.set_shape((clip_length,))  # Set fixed shape for batching
    return audio


def create_dataset(clean_dir, noisy_dir, batch_size):
    """Create a dataset from clean and noisy directories."""
    clean_files = [os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith(".wav")]
    noisy_files = [os.path.join(noisy_dir, f) for f in os.listdir(noisy_dir) if f.endswith(".wav")]

    clean_ds = tf.data.Dataset.from_tensor_slices(clean_files).map(tf_load_audio)
    noisy_ds = tf.data.Dataset.from_tensor_slices(noisy_files).map(tf_load_audio)

    dataset = tf.data.Dataset.zip((noisy_ds, clean_ds))
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset


# Create datasets
train_dataset = create_dataset(train_noisy_dir, train_clean_dir, batch_size)
val_dataset = create_dataset(val_noisy_dir, val_clean_dir, batch_size)
test_dataset = create_dataset(test_noisy_dir, test_clean_dir, batch_size)

print("Datasets ready!")
