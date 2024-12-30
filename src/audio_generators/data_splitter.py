import os
import random
import shutil

# Paths to clean and noisy datasets
clean_signals_dir = "data/pure"
noisy_signals_dir = "data/mixed"

# Output directories for splits
output_dirs = {
    "train": {"clean": "data/splits/train/clean", "noisy": "data/splits/train/noisy"},
    "val": {"clean": "data/splits/val/clean", "noisy": "data/splits/val/noisy"},
    "test": {"clean": "data/splits/test/clean", "noisy": "data/splits/test/noisy"},
}

# Create directories for splits
for split, paths in output_dirs.items():
    for signal_type, path in paths.items():
        os.makedirs(path, exist_ok=True)

# Get clean and noisy files
clean_files = sorted([f for f in os.listdir(clean_signals_dir) if f.endswith(".wav")])
noisy_files = sorted([f for f in os.listdir(noisy_signals_dir) if f.endswith(".wav")])

# Ensure datasets match in size
print(f"Number of clean files: {len(clean_files)}")
print(f"Number of noisy files: {len(noisy_files)}")
assert len(clean_files) == len(noisy_files), "Mismatch between clean and noisy datasets."

# Shuffle and split
dataset_size = len(clean_files)
indices = list(range(dataset_size))
random.shuffle(indices)

train_end = int(0.8 * dataset_size)
val_end = int(0.9 * dataset_size)

splits = {
    "train": indices[:train_end],
    "val": indices[train_end:val_end],
    "test": indices[val_end:],
}

# Copy files to split directories
for split, split_indices in splits.items():
    for idx in split_indices:
        clean_file = clean_files[idx]
        noisy_file = noisy_files[idx]

        # Copy clean signal
        shutil.copy(
            os.path.join(clean_signals_dir, clean_file),
            os.path.join(output_dirs[split]["clean"], clean_file),
        )

        # Copy noisy signal
        shutil.copy(
            os.path.join(noisy_signals_dir, noisy_file),
            os.path.join(output_dirs[split]["noisy"], noisy_file),
        )

print("Dataset split completed:")
print(f"Train: {len(splits['train'])} samples")
print(f"Validation: {len(splits['val'])} samples")
print(f"Test: {len(splits['test'])} samples")
