import tensorflow as tf
from tensorflow.keras import layers, Model

def build_discriminator(input_shape):
    """
    Creates the discriminator model.
    Args:
        input_shape: Tuple, shape of the input audio (e.g., (clip_length, 1)).
    Returns:
        discriminator: A Keras Model representing the discriminator.
    """
    inputs = layers.Input(shape=input_shape)

    # Downsample
    x = layers.Conv1D(8, kernel_size=15, strides=2, padding="same")(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv1D(16, kernel_size=15, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv1D(32, kernel_size=15, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Global Average Pooling
    x = layers.GlobalAveragePooling1D()(x)

    # Fully Connected Layer
    x = layers.Dense(16, activation="relu")(x)

    # Output Layer with sigmoid activation
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return Model(inputs, outputs, name="Discriminator")

# Example usage
clip_length = 44100 * 2  # 2 seconds of audio at 44.1 kHz
discriminator = build_discriminator((clip_length, 1))
discriminator.summary()
