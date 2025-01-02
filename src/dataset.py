import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import utils  # Assuming this is where config() is defined

class AudioDataset(Dataset):
    """
    Dataset class for audio signals (clean and mixed signals).
    """
    
    def __init__(self, partition, augment=False):
        """
        Initialize the dataset, loading clean and mixed signals.
        
        Args:
            partition (str): The dataset partition ("train", "val").
            augment (bool): Whether to apply data augmentation.
        """
        super().__init__()
        
        if partition not in ["train", "validation"]:
            raise ValueError(f"Partition {partition} does not exist")
        
        np.random.seed(42)
        torch.manual_seed(42)
        random.seed(42)
        
        self.partition = partition
        self.augment = augment
        
        # Load the clean and mixed data
        self.clean, self.mixed = self._load_data()

    def __len__(self):
        """
        Return the size of the dataset.
        """
        return len(self.clean)

    def __getitem__(self, idx):
        """
        Return clean and mixed audio samples at index `idx`.
        """
        clean_data = torch.from_numpy(self.clean[idx]).float().unsqueeze(0)
        mixed_data = torch.from_numpy(self.mixed[idx]).float().unsqueeze(0)
        
        # Get the corresponding filenames
        mixed_file = self.mixed_files[idx]
        clean_file = self.clean_files[idx]
        
        return mixed_data, clean_data, mixed_file, clean_file

    def _load_data(self):
        """
        Load clean and mixed data from disk based on the partition.
        """
        print(f"Loading {self.partition} data...")

        # Define the directories for clean and mixed signals
        clean_path = os.path.join("data", self.partition, "clean")  # Clean data path
        mixed_path = os.path.join("data", self.partition, "mixed")  # Mixed data path

        # Prepare lists to store data
        clean_data, mixed_data = [], []

        # List all files in clean and mixed directories
        clean_files = sorted(os.listdir(clean_path))
        mixed_files = sorted(os.listdir(mixed_path))

        # Ensure that clean and mixed files correspond to each other by filename
        for clean_file, mixed_file in zip(clean_files, mixed_files):
            # Load clean and mixed signals (assuming they are stored as .npy files or another format)
            clean_signal = np.load(os.path.join(clean_path, clean_file))  # Modify if not .npy
            mixed_signal = np.load(os.path.join(mixed_path, mixed_file))  # Modify if not .npy

            clean_data.append(clean_signal)
            mixed_data.append(mixed_signal)

        self.mixed_files = mixed_files  # Store filenames for reference
        self.clean_files = clean_files  # Store filenames for reference

        return np.array(clean_data), np.array(mixed_data)



def get_train_val_test_loaders(batch_size, **kwargs):
    """
    Return DataLoaders for train and val splits for clean and mixed signals.
    """
    
    # Create train and val datasets for clean and mixed signals
    tr_dataset = AudioDataset(partition="train", **kwargs)
    va_dataset = AudioDataset(partition="validation", **kwargs)
    
    # Create DataLoaders for clean and mixed signals
    tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va_dataset, batch_size=batch_size, shuffle=False)

    return tr_loader, va_loader

if __name__ == "__main__":
    batch_size = 32
    tr_loader, val_loader = get_train_val_test_loaders(batch_size)
    
    # Example of how to use the DataLoader
    for mixed_data, clean_data, mixed_file, clean_file in tr_loader:
        print(f"\nClean data batch shape: {clean_data.shape}\n")
        print(f"Mixed data batch shape: {mixed_data.shape}\n")
        print(f"Processing Mixed Data File: {mixed_file} \n\n Clean Data File: {clean_file}\n")
        break  # Just checking the first batch

