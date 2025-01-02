'''
Generate dataset of sine waves and gaussian white noise
'''

import numpy as np
import os
from scipy.io.wavfile import read, write
from scipy.signal import resample
import random
import utils.utils as utils

def generate_bounds(audio, length):
    """
    Generates bounds for a random span of time within an audio clip
    Parameters:
        audio - 1d numpy array of audio data
        length - integer of the length of a given clip
    Returns:
        bounds - tuple of two integers, the start and end indices on the random clip
    """
    # Find the maximum starting time of the clip
    max_start = len(audio) - length

    # Generate a random starting index and corresponding end index
    rand_start = np.random.randint(max_start)
    rand_end = rand_start + length

    return (rand_start, rand_end)

def mix_audio(signal, noise, snr):
    """
    Mixes signal and noise audio inputs to given signal to noise ratio
    (credit: https://stackoverflow.com/questions/71915018/mix-second-audio-clip-at-specific-snr-to-original-audio-file-in-python)
    Parameters:
        signal - 1d numpy array of the signal audio data
        noise - 1d numpy array of the noise's audio data
        snr - float of the signal to noise ratio in decibels
    Returns:
        bounds - tuple of two integers, the start and end indices on the random clip
    """
    # this is important if loading resulted in
    # uint8 or uint16 types, because it would cause overflow
    # when squaring and calculating mean
    noise = noise.astype(np.float32)
    signal = signal.astype(np.float32)

    # get the initial energy for reference
    signal_energy = np.mean(signal**2)
    noise_energy = np.mean(noise**2)
    # calculates the gain to be applied to the noise
    # to achieve the given SNR
    g = np.sqrt(10.0 ** (-snr/10) * signal_energy / noise_energy)

    # Assumes signal and noise to be decorrelated
    # and calculate (a, b) such that energy of
    # a*signal + b*noise matches the energy of the input signal
    a = np.sqrt(1 / (1 + g**2))
    b = np.sqrt(g**2 / (1 + g**2))
    # mix the signals
    return a * signal + b * noise

def generate_audio_combinations(bird_sounds, ambient_noise, sample_rate, clip_length=5, num_clips=1, snr=0.0):
    """
    Combines random samples of provided bird sounds and ambient noise into specified amount of clips 
    (with given length and signal to noise ratio)
    parameters:
        bird_sounds - 1d numpy array of sound data
        ambient_noise - 1d numpy array of ambient noise data
        sample_rate - integer, sample rate of given data (assumed to be the same for each audio data)
        clip_length - integer, the length of the clip (in seconds) that the user wants from the output
        num_clips - integer, the number of clips that the user wants outputted
        snr - float, signal to noise ratio (in decibels)
    returns:
        clips - numpy array of the data of the combined audio clips
        originals - numpy array of tuples of the two slices of data (bird, ambient)
    """
    # Find length of clip in terms of samples (depending on size of data, sample rate, and clip length)
    length = min(clip_length*sample_rate, len(bird_sounds), len(ambient_noise))

    # Create array, iterate as many times as user wants
    clips = []
    bird_original = []
    noise_original = []

    for i in range(num_clips):
        # Get the bounds for each random clip (of equal length)
        bird_clip_start, bird_clip_end = generate_bounds(bird_sounds, length)
        amb_clip_start, amb_clip_end = generate_bounds(ambient_noise, length)

        # Get audio clips from the bounds
        bird_clip = bird_sounds[bird_clip_start:bird_clip_end]
        amb_clip = ambient_noise[amb_clip_start:amb_clip_end]

        # Combine the signals based on the signal to noise ratio
        combined = mix_audio(bird_clip, amb_clip, snr)

        # Keep track of the results
        clips.append(combined)
        bird_original.append(bird_clip)
        noise_original.append(amb_clip)

    return np.array(clips), np.array(bird_original), np.array(noise_original)

def resample_and_write(signal, duration, i, output_dir="data/resampled") :
    # Resample to 44100 Hz for playback
    target_sample_rate = 44100  # or 48000 Hz if you prefer
    resampled_noise = resample(signal, int(target_sample_rate * duration))  # Resample to target rate
    
    # Convert to 16-bit PCM
    resampled_noise_int16 = np.int16(resampled_noise / np.max(np.abs(resampled_noise)) * 32767)
    
    # Save to a .wav file
    output_file = os.path.join(output_dir, f"resampled_{i+1:04d}.wav")
    write(output_file, target_sample_rate, resampled_noise_int16)

def generate_gwn(output_dir, amplitude_std=1.0, duration_std=0, sample_rate=1024, num_files=1000) :
    """
    Gaussian White Noise Generator

    Args:
        amplitude_std: Standard deviation of random amplitude distribution. Originally
                       0 for no randomness.
        duration_std: Standard deviation of random duration distribution. Originally
                       0 for no randomness.
        sample_rate (int): Sample rate at which GWN will be generated
        num_files (int): Number of GWN signals to generate.
        output_dir: Location where GWN signals will be stored.
        
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate num_files data of GWN
    for i in range(num_files):
        # Randomly sample duration
        duration = np.random.normal(loc=1.0, scale=duration_std)
        # Generate GWN
        noise = np.random.normal(0.0, amplitude_std, int(sample_rate * duration))
        # Save to a .wav file
        output_file = os.path.join(output_dir, f"noise_{i+1:04d}.npy")
        np.save(output_file, noise)
        #resample_and_write(noise, duration, i)

    print(f"Generated {num_files} noise signals in '{output_dir}' directory.")


def generate_sines(
        output_dir,
        low_freq=10,
        high_freq=256,
        num_components=5,
        amplitude_std=0, 
        duration_std=0,
        sample_rate=1024,
        num_files=1000,
        ) :
    """
    Sine-Wave Signal Generator

    Args:
        low_freq: Min frequency
        high_freq: Max frequency
        num_components: Number of sinusoidal components within signal
        amplitude_std: Standard deviation of random amplitude distribution. Originally
                       0 for no randomness.
        duration_std: Standard deviation of random duration distribution. Originally
                       0 for no randomness.
        sample_rate (int): Sample rate at which sine waves will be generated
        num_files (int): Number of sine wave signals to generate.
        output_dir: Location where GWN signals will be stored.

    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_files):
        # Initialize signal
        signal = np.zeros(int(sample_rate * 1), dtype=np.float32)
        # Generate time points
        duration = np.random.normal(loc=1, scale=duration_std)
        time = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        for j in range(num_components) :
            amplitude = np.random.normal(loc=1.0, scale=amplitude_std)
            frequency = np.random.uniform(low=low_freq, high=high_freq)
            # Generate sine wave
            component = amplitude * np.sin(2 * np.pi * frequency * time)
            signal += component
        
        # Amplitude is num_components times more than desired, simply normalize back 
        signal /= num_components
        # Save to a .wav file
        output_file = os.path.join(output_dir, f"sine_wave_{i+1:04d}.npy")
        np.save(output_file, signal)
        #resample_and_write(signal, duration, i)

    print(f"Generated {num_files} sine signals in '{output_dir}' directory.")



def generate_mixed_signals(
    noise_signals_dir,
    clean_signals_dir,
    mixed_signals_dir
) : 
    # Ensure the mixed directory exists
    os.makedirs(mixed_signals_dir, exist_ok=True)

    # TODO: Utilize varying SNRs
    snr = [-5, -3, -1, 0, 2, 4, 6]
    
    # Get list of clean and noise files
    clean_files = [f for f in os.listdir(clean_signals_dir) if f.endswith(".npy")]
    noise_files = [f for f in os.listdir(noise_signals_dir) if f.endswith(".npy")]

    i = 0
    # Mixing process
    for clean_file in clean_files:
        # Read clean signal
        clean_path = os.path.join(clean_signals_dir, clean_file)
        clean_signal = np.load(clean_path)

        # Randomly select a noise file
        noise_file = random.choice(noise_files)
        noise_path = os.path.join(noise_signals_dir, noise_file)
        noise_signal = np.load(noise_path)

        # Mix the audio
        mixed_signal = mix_audio(clean_signal, noise_signal, snr=snr[6])

        # Save the mixed signal
        mixed_file_name = f"mixed_{os.path.splitext(clean_file)[0]}_{os.path.splitext(noise_file)[0]}.npy"
        mixed_path = os.path.join(mixed_signals_dir, mixed_file_name)
        np.save(mixed_path, mixed_signal)
        resample_and_write(mixed_signal, 1.0, i)
        i += 1

    print(f"{len(clean_files)} mixed audio files saved in '{mixed_signals_dir}'.")



def main():

    kinds = ["/noise", "/clean", "/mixed"]
    splits = ["data/train", "data/validation"]

    directories = []

    for split in splits :
        for kind in kinds :
            # Combine the split and directory into a new path
            directories.append(split + kind)

    sample_rate=1024
    
    total_signals = 285
    tr_split, val_split = .9, .1
    
    # Wipe all directories
    for directory in directories:
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    for file in os.listdir("data/resampled"):
            file_path = os.path.join("data/resampled", file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    
     # Generate Gaussian White Noise
    generate_gwn(
        output_dir=directories[0],
        amplitude_std=0.5,
        duration_std=0,
        sample_rate=sample_rate,
        num_files=int(total_signals * tr_split),
    )
    generate_gwn(
        output_dir=directories[3],
        amplitude_std=0.5,
        duration_std=0,
        sample_rate=sample_rate,
        num_files=int(total_signals * val_split),
    )

    # Generate Sinusoidal Compositions
    generate_sines(
        output_dir=directories[1],
        low_freq=20,
        high_freq=40,
        num_components=1,
        amplitude_std=0,
        duration_std=0,
        sample_rate=sample_rate,
        num_files=int(total_signals * tr_split),
    )
    generate_sines(
        output_dir=directories[4],
        low_freq=20,
        high_freq=40,
        num_components=1,
        amplitude_std=0,
        duration_std=0,
        sample_rate=sample_rate,
        num_files=int(total_signals * val_split),
    )

    # Generate Mixed (GWN + Sines) Signals
    generate_mixed_signals(
        noise_signals_dir=directories[0],
        clean_signals_dir=directories[1],
        mixed_signals_dir=directories[2]
    )
    generate_mixed_signals(
        noise_signals_dir=directories[3], 
        clean_signals_dir=directories[4], 
        mixed_signals_dir=directories[5]
    )

    # Generate graphical displays of random signals from each directory
    time = np.linspace(0, 1, int(sample_rate * 1), endpoint=False)
    for directory in directories:
        numpy_files = [f for f in os.listdir(directory) if f.endswith(".npy")]
        if numpy_files:
            # Select a random .wav file
            random_file = random.choice(numpy_files)
            file_path = os.path.join(directory, random_file)
            # Read the audio file
            signal = np.load(file_path)
            # Display the signal using the plot_signal function
            name = f"Random Signal from {directory}"
            utils.plot_signal(time, signal, name)
            print(f"Displayed: {random_file} from {directory}")
        else:
            print(f"No .npy files found in {directory}")


if __name__ == "__main__":
    main()