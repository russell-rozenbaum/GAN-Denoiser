"""
Utility functions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import torch


def hold_training_plot():
    """
    Keep the program alive to display the training plot
    """
    plt.ioff()
    plt.show()

def save_training_plot(save_path="images/performance_plot.png"):
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
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))  
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

    axes[1, 2].set_title("Discriminator AUROC")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("AUROC")

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

    if model_to_eval == "discriminator":
        # AUROC Subplot
        axes[row, 2].clear()
        axes[row, 2].plot(x_data, train_auroc_list, 'b-o', label='Train AUROC')
        axes[row, 2].plot(x_data, val_auroc_list,   'g-o', label='Val AUROC')
        axes[row, 2].set_title(f"{title_prefix} AUROC")
        axes[row, 2].set_xlabel("Epoch")
        axes[row, 2].set_ylabel("AUROC")
        axes[row, 2].legend()

    plt.tight_layout()
    plt.pause(0.001)

def get_mel_spectrogram(file_path, n_mels=128, fmax=8000):
    y, sr = librosa.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

def plot_signal(time, amplitude, name) :
    plt.figure()
    plt.plot(time[:100], amplitude[:100])
    plt.title(name)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

def plot_signal_grid(signals_dict, time, sample_window=100, title="Generated Signals"):
    """
    Plot multiple signals in a 2x3 grid and display generation parameters in the title.
    
    Args:
        signals_dict (dict): Dictionary with keys like 'train/noise', 'train/mixed', etc.
        time (numpy.array): Time array for x-axis.
        sample_window (int): Number of samples to show in plot.
        title (str): Title for the entire plot.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
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
    plt.show()

def plot_denoising_results(generator, val_loader, title, num_examples=2, sample_window=100):
    """
    Plot mixed signals, their corresponding clean signals, and denoised signals.
    
    Args:
        generator: The generator model for denoising.
        val_loader: DataLoader containing validation data.
        num_examples: Number of examples to plot.
        sample_window: Number of samples to display in each plot.
    """
    import random

    # Initialize the plot grid
    fig, axes = plt.subplots(num_examples, 3, figsize=(15, 6))
    col_titles = ['Clean Signal', 'Mixed Signal', 'Denoised Signal']
    
    # Iterate over the number of examples
    for i in range(num_examples):
        # Fetch a batch of data from the validation loader
        batch = next(iter(val_loader))
        mixed_data, clean_data, _, _ = batch 

        # Choose a random index from the batch
        random_idx = random.randint(0, mixed_data.shape[0] - 1)
        random_mixed_signal = mixed_data[random_idx].squeeze().detach().cpu().numpy()
        random_clean_signal = clean_data[random_idx].squeeze().detach().cpu().numpy()

        # Generate the denoised signal using the generator
        with torch.no_grad():
            generated_signal = generator(torch.tensor(random_mixed_signal[None, None, :], dtype=torch.float32)).cpu().numpy().squeeze()

        # Create a time array for plotting
        time = np.linspace(0, 1, len(random_mixed_signal), endpoint=False)

        # Plot each signal in its respective column
        axes[i, 0].plot(time[:sample_window], random_clean_signal[:sample_window])
        axes[i, 0].set_title(col_titles[0])
        axes[i, 0].set_ylabel(f"Example {i + 1}")
        axes[i, 0].grid(alpha=0.3)

        axes[i, 1].plot(time[:sample_window], random_mixed_signal[:sample_window])
        axes[i, 1].set_title(col_titles[1])
        axes[i, 1].grid(alpha=0.3)

        axes[i, 2].plot(time[:sample_window], generated_signal[:sample_window])
        axes[i, 2].set_title(col_titles[2])
        axes[i, 2].grid(alpha=0.3)

    # Add global title and adjust layout
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


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