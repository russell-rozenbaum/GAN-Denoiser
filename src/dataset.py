import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import utils  # Assuming this is where config() is defined

class AudioDataset(Dataset):
    """Dataset class for audio signals (clean and mixed signals)."""
    
    def __init__(self, partition, task="target", augment=False):
        """Initialize the dataset, loading clean and mixed signals.
        
        Args:
            partition (str): The dataset partition ("train", "val").
            task (str): The task to handle ("target" for clean, "source" for mixed).
            augment (bool): Whether to apply data augmentation.
        """
        super().__init__()
        
        if partition not in ["train", "validation"]:
            raise ValueError(f"Partition {partition} does not exist")
        
        np.random.seed(42)
        torch.manual_seed(42)
        random.seed(42)
        
        self.partition = partition
        self.task = task
        self.augment = augment
        
        # Load the clean and mixed data
        self.clean, self.mixed = self._load_data()

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.clean)

    def __getitem__(self, idx):
        """Return clean or mixed audio sample at index `idx`."""
        if self.task == "target":
            return torch.from_numpy(self.clean[idx]).float().unsqueeze(0)
        elif self.task == "source":
            return torch.from_numpy(self.mixed[idx]).float().unsqueeze(0)
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def _load_data(self):
        """Load clean and mixed data from disk based on the partition."""
        print(f"Loading {self.partition} data...")

        # Define the directories for clean and mixed signals
        clean_path = os.path.join("data", self.partition, "clean")  # Clean data path
        mixed_path = os.path.join("data", self.partition, "mixed")  # Mixed data path

        # Prepare lists to store data
        clean_data, mixed_data = [], []

        # List all files in clean and mixed directories
        clean_files = os.listdir(clean_path)
        mixed_files = os.listdir(mixed_path)

        # Ensure that clean and mixed files correspond to each other
        for clean_file, mixed_file in zip(clean_files, mixed_files):
            # Load clean and mixed signals (assuming they are stored as .npy files or another format)
            clean_signal = np.load(os.path.join(clean_path, clean_file))  # Modify if not .npy
            mixed_signal = np.load(os.path.join(mixed_path, mixed_file))  # Modify if not .npy

            clean_data.append(clean_signal)
            mixed_data.append(mixed_signal)

        return np.array(clean_data), np.array(mixed_data)


def get_train_val_test_loaders(batch_size, **kwargs):
    """Return DataLoaders for train and val splits for clean and mixed signals."""
    
    # Create train and val datasets for clean and mixed signals
    tr_clean = AudioDataset(partition="train", task="target", **kwargs)
    tr_mixed = AudioDataset(partition="train", task="source", **kwargs)
    va_clean = AudioDataset(partition="validation", task="target", **kwargs)
    va_mixed = AudioDataset(partition="validation", task="source", **kwargs)
    
    # Create DataLoaders for clean and mixed signals
    tr_clean_loader = DataLoader(tr_clean, batch_size=batch_size, shuffle=True)
    tr_mixed_loader = DataLoader(tr_mixed, batch_size=batch_size, shuffle=True)
    va_clean_loader = DataLoader(va_clean, batch_size=batch_size, shuffle=False)
    va_mixed_loader = DataLoader(va_mixed, batch_size=batch_size, shuffle=False)

    return tr_clean_loader, tr_mixed_loader, va_clean_loader, va_mixed_loader


if __name__ == "__main__":
    batch_size = 500
    tr_clean_loader, tr_mixed_loader, va_clean_loader, va_mixed_loader = get_train_val_test_loaders(batch_size)
    
    # Example of how to use the DataLoaders
    for clean_data, mixed_data in zip(tr_clean_loader, tr_mixed_loader):
        print(f"Clean data batch shape: {clean_data.shape}")
        print(f"Mixed data batch shape: {mixed_data.shape}")
        break  # Just checking the first batch
