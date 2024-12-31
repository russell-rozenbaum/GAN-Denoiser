import tensorflow as tf
from tensorflow.keras import Model
import generator
import discriminator

def build_gan(generator, discriminator):
    """
    Combines the generator and discriminator into a GAN model.
    Args:
        generator: The generator model.
        discriminator: The discriminator model.
    Returns:
        gan: A Keras Model representing the GAN.
    """
    # Ensure the discriminator is not trainable when training the GAN
    discriminator.trainable = False

    # Input: Noisy audio
    gan_input = tf.keras.Input(shape=generator.input_shape[1:])

    # Generator: Produces denoised audio
    generated_audio = generator(gan_input)

    # Discriminator: Classifies the generator's output
    gan_output = discriminator(generated_audio)

    # Full GAN model
    gan = Model(gan_input, gan_output, name="GAN")

    return gan

# Build generator and discriminator
clip_length = 44100 * 2  # 2 seconds of audio at 44.1 kHz
generator = build_generator((clip_length, 1))
discriminator = build_discriminator((clip_length, 1))

# Build GAN
gan = build_gan(generator, discriminator)
gan.summary()
