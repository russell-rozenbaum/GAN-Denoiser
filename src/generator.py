import tensorflow as tf
from tensorflow.keras import layers, Model

def build_generator(input_shape):
    """
    Creates the generator (aka denoiser) model.
    Args:
        input_shape: Tuple, shape of the input noisy audio (e.g., (clip_length, 1) where clip_length is the total number of samples).
    Returns:
        generator: A Keras Model representing the generator.
    """

    inputs = layers.Input(shape=input_shape)

    # Encoder: Downsample
    # conv1d
    x = layers.Conv1D(64, kernel_size=15, strides=2, padding="same", activation="relu")(inputs)
    # conv1d_1
    x = layers.Conv1D(128, kernel_size=15, strides=2, padding="same", activation="relu")(x)
    # conv1d_2
    x = layers.Conv1D(256, kernel_size=15, strides=1, padding="same", activation="relu")(x)

    # Decoder: Upsample
    # conv1d_transpose
    x = layers.Conv1DTranspose(128, kernel_size=15, strides=2, padding="same", activation="relu")(x)
    # conv1d_transpose_1
    x = layers.Conv1DTranspose(64, kernel_size=15, strides=2, padding="same", activation="relu")(x)

    # Output layer
    outputs = layers.Conv1D(1, kernel_size=15, strides=1, padding="same", activation="sigmoid")(x)

    return Model(inputs, outputs, name="Generator")

# Example usage
# 2 seconds of audio at 44100 samples/sec
clip_length = 44100 * 2  
generator = build_generator((clip_length, 1))
generator.summary()
