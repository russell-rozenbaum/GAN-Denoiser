import tensorflow as tf
import numpy as np
from scipy.io.wavfile import read, write  # Import read and write from scipy for reading and saving wav files

def test_generator_on_audio(generator_model_path, noisy_audio_path, output_audio_path, noisy_audio_output_path):
    # Load the trained generator model (saved as .keras)
    generator = tf.keras.models.load_model(generator_model_path)

    # Load the noisy audio file using scipy.io.wavfile.read
    sample_rate, noisy_audio = read(noisy_audio_path)  # `read` returns sample rate and audio data

    # Normalize the noisy audio to range [-1, 1] for processing
    noisy_audio_normalized = noisy_audio.astype(np.float32) / np.max(np.abs(noisy_audio))  # Normalize to [-1, 1]

    # Generate the denoised audio using the trained generator
    denoised_audio = generator.predict(noisy_audio_normalized.reshape(-1, 1))

    # Inspect the generated output tensor values
    print("Generated audio tensor statistics:")
    print(f"Min value: {np.min(denoised_audio)}")
    print(f"Max value: {np.max(denoised_audio)}")
    print(f"Mean value: {np.mean(denoised_audio)}")
    print(f"Standard deviation: {np.std(denoised_audio)}")

    # Amplify the denoised audio output (you can adjust the factor as needed)
    denoised_audio = denoised_audio * 100  # Amplify the output by a factor (tune as necessary)

    # Inspect the generated output tensor values
    print("Generated audio tensor statistics:")
    print(f"Min value: {np.min(denoised_audio)}")
    print(f"Max value: {np.max(denoised_audio)}")
    print(f"Mean value: {np.mean(denoised_audio)}")
    print(f"Standard deviation: {np.std(denoised_audio)}")

    # Ensure the denoised audio is scaled to the proper range ([-1, 1])
    denoised_audio = np.clip(denoised_audio, -1.0, 1.0)  # Clip the values to [-1, 1] range

    # Denormalize the output audio if necessary (reverse the normalization step)
    denoised_audio = denoised_audio * np.max(np.abs(noisy_audio))  # Scale back to the original range

    # Clip the audio values to fit the range of int16 ([-32768, 32767])
    denoised_audio = np.clip(denoised_audio, -32768, 32767)

    # Save the noisy audio as is (to listen to the original)
    write(noisy_audio_output_path, sample_rate, noisy_audio.astype(np.int16))  # Save noisy audio as int16

    # Save the denoised audio to an output file using scipy.io.wavfile.write
    write(output_audio_path, sample_rate, denoised_audio.astype(np.int16))  # Convert to int16 and save

    print(f"Noisy audio saved to {noisy_audio_output_path}")
    print(f"Denoised audio saved to {output_audio_path}")

# Correct path to the uploaded noisy audio file
noisy_audio_path = "/Users/russellrozenbaum/Desktop/KwF/GAN Denoiser/data/mixed/mixed_sine_wave_0001_noise_0029.wav"  # Correct path
output_audio_path = "denoised_audio_file.wav"
noisy_audio_output_path = "original_noisy_audio.wav"

# Test the model
generator_model_path = "generator_model_after_100_epochs.keras"  # .keras model path
test_generator_on_audio(generator_model_path, noisy_audio_path, output_audio_path, noisy_audio_output_path)
