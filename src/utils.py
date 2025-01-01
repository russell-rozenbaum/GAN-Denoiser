"""
Utility functions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import librosa


def hold_training_plot():
    """
    Keep the program alive to display the training plot
    """
    plt.ioff()
    plt.show()


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

    axes[0, 2].set_title("Generator AUROC")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("AUROC")

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



def update_training_plot(axes, epoch, stats, model_to_eval):
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

    # AUROC Subplot
    axes[row, 2].clear()
    axes[row, 2].plot(x_data, train_auroc_list, 'b-o', label='Train AUROC')
    axes[row, 2].plot(x_data, val_auroc_list,   'g-o', label='Val AUROC')
    axes[row, 2].set_title(f"{title_prefix} AUROC")
    axes[row, 2].set_xlabel("Epoch")
    axes[row, 2].set_ylabel("AUROC")
    axes[row, 2].legend()

    plt.tight_layout()
    plt.pause(0.0001)


def save_cnn_training_plot(patience,use_augment):
    """Save the training plot to a file."""
    if use_augment:
        plt.savefig(f"cnn_training_plot_patience={patience}_augmented.png", dpi=200)
    else:
        plt.savefig(f"cnn_training_plot_patience={patience}.png", dpi=200)

def get_mel_spectrogram(file_path, n_mels=128, fmax=8000):
    y, sr = librosa.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

def plot_signal(time, amplitude, name) :
    plt.plot(time[:300], amplitude[:300])
    plt.title(name)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()