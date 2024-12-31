"""
Utility functions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import librosa


def config(attr):
    """
    Retrieves the queried attribute value from the config file. Loads the
    config file on first call.
    """
    if not hasattr(config, "config"):
        with open("config.json") as f:
            config.config = eval(f.read())
    node = config.config
    for part in attr.split("."):
        node = node[part]
    return node


def denormalize_image(image):
    """ Rescale the image's color space from (min, max) to (0, 1) """
    ptp = np.max(image, axis=(0, 1)) - np.min(image, axis=(0, 1))
    return (image - np.min(image, axis=(0, 1))) / ptp


def hold_training_plot():
    """
    Keep the program alive to display the training plot
    """
    plt.ioff()
    plt.show()


def log_training(epoch, stats):
    """Print the train, validation, test accuracy/loss/auroc.

    args:
    
    stats (list): A cumulative list to store the model generator loss and auroc, and discriminator loss and auroc for every epoch.
            Usage: stats[epoch][0] = validation gen loss, stats[epoch][1] = validation gen auroc,
                   stats[epoch][2] = validation disc loss, stats[epoch][3] = validation disc auroc
                   etc for loss and test
                  
    epoch (int): The current epoch number.
    
    Note: Test accuracy is optional and will only be logged if stats is length 9.
    """
    include_train = len(stats[-1]) / 4 == 4
    splits = ["Validation", "Train", "Test"]
    metrics = ["Gen Loss", "Gen AUROC", "Disc Loss", "Disc AUROC"]
    print("Epoch {}".format(epoch))
    for j, split in enumerate(splits):
        for i, metric in enumerate(metrics):
            idx = len(metrics) * j + i
            # Checking if stats contains the test metrics. If not, ignore. 
            if idx >= len(stats[-1]):
                continue
            print(f"\t{split} {metric}:{round(stats[-1][idx],4)}")


def make_training_plot(name="CNN Training"):
    """Set up an interactive matplotlib graph to log metrics during training."""
    plt.ion()
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    plt.suptitle(name)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Gen Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Gen AUROC")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Disc Loss")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("Disc AUROC")

    return axes


def update_training_plot(axes, epoch, stats):
    """Update the training plot with a new data point for loss and AUROC."""
    metrics = ["Gen Loss", "Gen AUROC", "Disc Loss", "Disc AUROC"]
    splits = ["Validation", "Train", "Test"]
    colors = ["r", "b", "g"]

    for i, metric in enumerate(metrics):
        for j, split in enumerate(splits):
            idx = len(metrics) * j + i
            if idx >= len(stats[-1]):
                continue

            # Ensure the plot updates with correct values for the specified split
            axes[i].plot(
                range(1, epoch + 1),
                [stat[idx] for stat in stats],
                linestyle="--",
                marker="o",
                color=colors[j % len(colors)],
            )
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel(metric)
        axes[i].legend(splits)

    plt.pause(0.0001)  # Make sure the plot updates without stalling

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