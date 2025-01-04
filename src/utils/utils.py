"""
Utility functions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import torch
from scipy.io.wavfile import write
from scipy.signal import resample
from scipy import signal


def hold_plot():
    """
    Keep the program alive to display the training plot
    """
    plt.ioff()
    plt.show()

def save_plot(save_path):
    """
    Save the current training plot and keep the program alive to display it.
    
    Args:
        save_path (str): Path to save the performance plot.
    """
    # Save the current figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists
    plt.savefig(save_path, format="png", dpi=300)
    print(f"Plot saved to {save_path}")

def close_plot():
    """
    Keep the program alive to display the training plot
    """
    plt.close('all')

def log_training(epoch, stats, model_to_eval):
    """
    Print the train and validation accuracy/loss/auroc.
    """
    epoch_stats = stats[epoch]
    print(f"Stats for {model_to_eval} at epoch {epoch}")
    # Helper function to safely print a key if it exists
    def maybe_print(key, label):
        if key in epoch_stats:
            print(f"\t{label}: {epoch_stats[key]:.4f}")
    # Print train metrics
    maybe_print("train_loss", "Train Loss")
    maybe_print("train_acc", "Train Accuracy")
    maybe_print("train_auroc", "Train AUROC")
    # Print val metrics
    maybe_print("val_loss", "Val Loss")
    maybe_print("val_acc", "Val Accuracy")
    maybe_print("val_auroc", "Val AUROC")


import matplotlib.pyplot as plt

def make_training_plot(name="GAN Training"):
    """Set up an interactive matplotlib figure with 6 subplots:
       3 metrics (loss, accuracy, auroc) for generator (row 0)
       3 metrics (loss, accuracy, auroc) for discriminator (row 1)
       Each subplot shows train vs. val curves in different colors.
    """
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(18, 8))  
    # 2 rows (Generator, Discriminator) x 3 columns (Loss, Accuracy, AUROC)

    # Give an overall title
    plt.suptitle(name, fontsize=16)

    # --- Row 0: Generator ---
    axes[0, 0].set_title("Generator Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")

    axes[0, 1].set_title("Generator Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")

    # --- Row 1: Discriminator ---
    axes[1, 0].set_title("Discriminator Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")

    axes[1, 1].set_title("Discriminator Accuracy")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")

    '''
    axes[1, 2].set_title("Discriminator AUROC")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("AUROC")
    '''

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
    return axes



def update_training_plot(axes, stats, model_to_eval):
    """
    Args:
        axes: a 2D array of subplots (2 rows, 3 columns)
            row 0 => Generator subplots
            row 1 => Discriminator subplots
        stats: list of dict with keys:
            ["train_loss", "val_loss", "train_acc", "val_acc", "train_auroc", "val_auroc"]
        model_to_eval: "generator" or "discriminator"
    """
    x_data = range(1, len(stats) + 1)

    # Extract series from stats
    train_loss_list = [s["train_loss"] for s in stats]
    val_loss_list   = [s["val_loss"]   for s in stats]
    train_acc_list  = [s["train_acc"]  for s in stats]
    val_acc_list    = [s["val_acc"]    for s in stats]
    if model_to_eval == "discriminator":
        train_auroc_list= [s["train_auroc"]for s in stats]
        val_auroc_list  = [s["val_auroc"]  for s in stats]

    # Decide which row to update
    if model_to_eval == "generator":
        row = 0  # generator row
        title_prefix = "Generator"
    else:
        row = 1  # discriminator row
        title_prefix = "Discriminator"

    # --- Clear and re-plot the series in each of the 3 subplots in the row ---
    # Loss Subplot
    axes[row, 0].clear()
    axes[row, 0].plot(x_data, train_loss_list, 'b-o', label='Train Loss')
    axes[row, 0].plot(x_data, val_loss_list,   'g-o', label='Val Loss')
    axes[row, 0].set_title(f"{title_prefix} Loss")
    axes[row, 0].set_xlabel("Epoch")
    axes[row, 0].set_ylabel("Loss")
    axes[row, 0].legend()

    # Accuracy Subplot
    axes[row, 1].clear()
    axes[row, 1].plot(x_data, train_acc_list, 'b-o', label='Train Acc')
    axes[row, 1].plot(x_data, val_acc_list,   'g-o', label='Val Acc')
    axes[row, 1].set_title(f"{title_prefix} Accuracy")
    axes[row, 1].set_xlabel("Epoch")
    axes[row, 1].set_ylabel("Accuracy")
    axes[row, 1].legend()

    '''
    if model_to_eval == "discriminator":
        # AUROC Subplot
        axes[row, 2].clear()
        axes[row, 2].plot(x_data, train_auroc_list, 'b-o', label='Train AUROC')
        axes[row, 2].plot(x_data, val_auroc_list,   'g-o', label='Val AUROC')
        axes[row, 2].set_title(f"{title_prefix} AUROC")
        axes[row, 2].set_xlabel("Epoch")
        axes[row, 2].set_ylabel("AUROC")
        axes[row, 2].legend()
    '''

    plt.tight_layout()
    plt.pause(0.001)

def get_mel_spectrogram(file_path, n_mels=128, fmax=8000):
    y, sr = librosa.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

def plot_signal_grid(signals_dict, time, sample_window=120, title="Generated Signals"):
    """
    Plot multiple signals in a 2x3 grid and display generation parameters in the title.
    
    Args:
        signals_dict (dict): Dictionary with keys like 'train/noise', 'train/mixed', etc.
        time (numpy.array): Time array for x-axis.
        sample_window (int): Number of samples to show in plot.
        title (str): Title for the entire plot.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 6))
    col_titles = ['Clean', 'Noise', 'Mixed']
    row_titles = ['Training', 'Validation']
    
    for i, split in enumerate(['train', 'validation']):
        for j, signal_type in enumerate(['clean', 'noise', 'mixed']):
            key = f"{split}/{signal_type}"
            if key in signals_dict:
                signal = signals_dict[key]
                axes[i, j].plot(time[:sample_window], signal[:sample_window])
                axes[i, j].set_title(f"{row_titles[i]} {col_titles[j]}")
                axes[i, j].grid(True, alpha=0.3)
                if j == 0:
                    axes[i, j].set_ylabel("Amplitude")
                if i == 1:
                    axes[i, j].set_xlabel("Time (s)")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    plt.ion()  # Turn on interactive mode
    plt.pause(0.001)  # Small pause to render


def plot_denoising_results_against_filter(generator, val_loader, title, num_examples=2, sample_window=120):
    """
    Plot mixed signals, clean signals, denoised signals, and lowpass filtered signals.
    
    Args:
        generator: The generator model for denoising
        val_loader: DataLoader containing validation data
        title (str): Title for the plot
        num_examples (int): Number of examples to plot
        sample_window (int): Number of samples to display in each plot
    """
    # Initialize the plot grid - now with 4 columns
    fig, axes = plt.subplots(num_examples, 4, figsize=(20, 6))
    col_titles = ['Clean Signal', 'Mixed Signal', 'Denoised Signal', 'Lowpass Filtered']
    
    for i in range(num_examples):
        # Fetch a batch of data
        batch = next(iter(val_loader))
        mixed_data, clean_data, _, _ = batch
        
        # Choose a random index from the batch
        random_idx = np.random.randint(0, mixed_data.shape[0])
        random_mixed_signal = mixed_data[random_idx].squeeze().detach().cpu().numpy()
        random_clean_signal = clean_data[random_idx].squeeze().detach().cpu().numpy()
        
        # Generate denoised signal
        with torch.no_grad():
            generated_signal = generator(
                torch.tensor(random_mixed_signal[None, None, :], dtype=torch.float32)
            ).cpu().numpy().squeeze()
        
        # Apply lowpass filter
        filtered_signal = apply_lowpass_filter(random_mixed_signal)
        
        # Create time array for plotting
        time = np.linspace(0, 1, len(random_mixed_signal), endpoint=False)
        
        # Save audio files
        resample_and_write(signal=random_clean_signal, name=f"clean_{i+1:04d}")
        resample_and_write(signal=random_mixed_signal, name=f"mixed_{i+1:04d}")
        resample_and_write(signal=generated_signal, name=f"denoised_{i+1:04d}")
        resample_and_write(signal=filtered_signal, name=f"filtered_{i+1:04d}")
        
        # Plot all signals
        for j, signal in enumerate([random_clean_signal, random_mixed_signal, 
                                  generated_signal, filtered_signal]):
            axes[i, j].plot(time[:sample_window], signal[:sample_window])
            axes[i, j].set_title(col_titles[j])
            if j == 0:
                axes[i, j].set_ylabel(f"Example {i + 1}")
            axes[i, j].grid(alpha=0.3)
            if i == num_examples - 1:
                axes[i, j].set_xlabel("Time (s)")
    
    # Add global title and adjust layout
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    plt.ion()
    plt.pause(0.001)

def plot_generator_against_lowpass(generator, val_loader, num_examples=2, sample_window=120):
    """
    Plot the frequency spectrum for mixed signals, clean signals, denoised signals, and lowpass filtered signals.
    
    Args:
        generator: The generator model for denoising
        val_loader: DataLoader containing validation data
        title (str): Title for the plot
        num_examples (int): Number of examples to plot
        sample_window (int): Number of samples to display in each plot
    """
    amp_titles = ['Clean Signal', 'Mixed Signal', 'Denoised Signal', 'Lowpass Filtered']
    freq_titles = ['Clean Spectrum', 'Mixed Spectrum', 'Denoised Spectrum', 'Lowpass Spectrum']

    for ex_idx in range(num_examples):
        # Fetch a batch of data
        batch = next(iter(val_loader))
        mixed_data, clean_data, _, _ = batch
        
        # Choose a random index from the batch
        random_idx = np.random.randint(0, mixed_data.shape[0])
        mixed_signal = mixed_data[random_idx].squeeze().detach().cpu().numpy()
        clean_signal = clean_data[random_idx].squeeze().detach().cpu().numpy()
        
        # Generate denoised signal
        with torch.no_grad():
            generated_signal = generator(
                torch.tensor(mixed_signal[None, None, :], dtype=torch.float32)
            ).cpu().numpy().squeeze()
        
        # Apply lowpass filter
        filtered_signal = apply_lowpass_filter(mixed_signal)
        
        # Create time array for plotting
        time = np.linspace(0, 1, len(mixed_signal), endpoint=False)
        
        # Save audio files
        resample_and_write(signal=clean_signal, name=f"clean_{ex_idx+1:04d}")
        resample_and_write(signal=mixed_signal, name=f"mixed_{ex_idx+1:04d}")
        resample_and_write(signal=generated_signal, name=f"denoised_{ex_idx+1:04d}")
        resample_and_write(signal=filtered_signal, name=f"filtered_{ex_idx+1:04d}")
        
        # Initialize the plot grid for this example - 2 rows, 4 columns (1 row for signals, 1 row for spectra)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Plot time-amplitude signals in the first row
        for col, signal in enumerate([clean_signal, mixed_signal, generated_signal, filtered_signal]):
            axes[0, col].plot(time[:sample_window], signal[:sample_window])
            axes[0, col].set_title(amp_titles[col])
            axes[0, col].grid(alpha=0.3)
             # Set y-axis label for amplitude
            axes[0, col].set_xlabel("Time (s)")
            if col == 0 :
                axes[0, col].set_ylabel("Amplitude")

        # Plot frequency spectra in the second row
        for col, signal in enumerate([clean_signal, mixed_signal, generated_signal, filtered_signal]):
            axes[1, col].plot(np.linspace(0, 128, len(signal)//2), 
                              np.abs(np.fft.fft(signal))[:len(signal)//2])
            axes[1, col].set_title(freq_titles[col])
            axes[1, col].grid(True, alpha=0.3)
            # Set y-axis label for magnitude
            axes[1, col].set_xlabel("Frequency (Hz)")
            if col == 0 :
                axes[1, col].set_ylabel("Magnitude")
        
        # Add global title and adjust layout
        fig.suptitle(f"Denoiser Against Lowpass Filter\nExample {ex_idx + 1}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        plt.show()
  
        fig.savefig(f"images/denoised_against_lowpass_{ex_idx + 1:02d}.png")

        plt.close(fig)


def normalize_signal(signal):
    """
    Normalize the input signal to be between -1 and 1.
    Args:
        signal (numpy.ndarray or torch.Tensor): Input signal with arbitrary amplitude values.
    Returns:
        numpy.ndarray or torch.Tensor: Normalized signal between -1 and 1.
    """
    min_val = signal.min()
    max_val = signal.max()
    normalized_signal = 4 * (signal - min_val) / (max_val - min_val) - 2
    
    return normalized_signal

def resample_and_write(signal, name, duration=1., output_dir="data/resampled") :
    # Resample to 44100 Hz for playback
    target_sample_rate = 44100  # or 48000 Hz if you prefer
    resampled_noise = resample(signal, int(target_sample_rate * duration))  # Resample to target rate
    
    # Convert to 16-bit PCM
    resampled_noise_int16 = np.int16(resampled_noise / np.max(np.abs(resampled_noise)) * 32767)
    
    # Save to a .wav file
    output_file = os.path.join(output_dir, f"resampled_{name}.wav")
    write(output_file, target_sample_rate, resampled_noise_int16)

    
def apply_lowpass_filter(audio_signal, sr=256, passband_freq=64, stopband_freq=100, 
                        passband_ripple=0.05, stopband_attenuation=60):
    """
    Apply a Butterworth lowpass filter to an audio signal.
    
    Args:
        audio_signal (numpy.ndarray): Input audio signal
        sr (int): Sampling rate in Hz
        passband_freq (float): Passband frequency in Hz
        stopband_freq (float): Stopband frequency in Hz
        passband_ripple (float): Maximum ripple allowed in the passband (dB)
        stopband_attenuation (float): Minimum attenuation required in the stopband (dB)
    
    Returns:
        numpy.ndarray: Filtered audio signal
    """
    nyquist = sr / 2
    
    # Normalize frequencies to Nyquist rate
    wp = passband_freq / nyquist
    ws = stopband_freq / nyquist
    
    # Calculate filter order and cutoff frequency
    order, wn = signal.buttord(wp, ws, passband_ripple, stopband_attenuation)
    
    # Create Butterworth filter
    b, a = signal.butter(order, wn, btype='low')
    
    # Apply zero-phase filtering
    filtered_signal = signal.filtfilt(b, a, audio_signal)
    
    return filtered_signal
    