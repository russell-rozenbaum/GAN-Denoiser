import numpy as np

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