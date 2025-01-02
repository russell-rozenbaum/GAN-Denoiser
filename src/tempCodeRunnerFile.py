        # Get paths for mixed and clean signals
        mixed_path = os.path.join(self.mixed_dir, self.mixed_files[idx])
        clean_path = os.path.join(self.clean_dir, self.clean_files[idx])
        
        # Load data (assuming numpy files)
        mixed_data = np.load(mixed_path)
        clean_data = np.load(clean_path)
        
        # Convert data to tensors if necessary
        mixed_data = torch.tensor(mixed_data, dtype=torch.float32)
        clean_data = torch.tensor(clean_data, dtype=torch.float32)
        
        # Return the data as tensors
        return mixed_data, clean_data