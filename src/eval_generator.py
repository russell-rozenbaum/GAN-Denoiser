import torch
import os
from dataset import get_train_val_test_loaders
from models.generator import DenoisingAE
import utils.utils as utils
import utils.generator_utils as gen_utils

def load_generator_from_checkpoint(checkpoint_path, signal_length=256):
    """Load generator from checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    
    # Extract generator parameters from the checkpoint
    generator = DenoisingAE(
        signal_length=signal_length,
        num_enc_filters=16,  # These parameters should match your training configuration
        num_dec_filters=32,
        num_downsamples=4,
        num_upsamples=4,
        gamma=3,
        rho=9,
    )
    
    generator.load_state_dict(checkpoint['generator_state_dict'])
    return generator, checkpoint['generator_stats']

def evaluate_generator(generator, val_loader, num_examples=5):
    """Evaluate generator and create plots"""
    generator.eval()
    
    # Create performance plots
    utils.plot_generator_against_lowpass(
        generator=generator,
        val_loader=val_loader,
        num_examples=num_examples,
    )
    
    # Calculate and print validation metrics
    total_val_loss = 0
    total_val_recon_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            clean_signals = batch['clean_signal']
            noisy_signals = batch['noisy_signal']
            
            reconstructed = generator(noisy_signals)
            
            val_loss = gen_utils.compute_loss(
                generator=generator,
                reconstructed=reconstructed,
                clean_signals=clean_signals,
                discriminator=None  # We don't need discriminator for evaluation
            )
            
            recon_loss = torch.nn.functional.mse_loss(reconstructed, clean_signals)
            
            total_val_loss += val_loss.item()
            total_val_recon_loss += recon_loss.item()
            num_batches += 1
    
    avg_val_loss = total_val_loss / num_batches
    avg_recon_loss = total_val_recon_loss / num_batches
    
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Reconstruction Loss: {avg_recon_loss:.4f}")
    
    return avg_val_loss, avg_recon_loss

def get_available_checkpoints():
    """Get list of available checkpoint epochs"""
    checkpoints = []
    if os.path.exists('checkpoints'):
        for file in os.listdir('checkpoints'):
            if file.startswith('checkpoint_epoch_') and file.endswith('.pt'):
                epoch = int(file.split('_')[-1].split('.')[0])
                checkpoints.append(epoch)
    return sorted(checkpoints)

def main():
    # Show available checkpoints
    available_epochs = get_available_checkpoints()
    if not available_epochs:
        print("No checkpoints found in the 'checkpoints' directory!")
        return
    
    print("\nAvailable checkpoint epochs:", available_epochs)
    
    # Get epoch from user
    while True:
        try:
            epoch = int(input("\nEnter the epoch number to evaluate: "))
            if epoch not in available_epochs:
                print(f"Epoch {epoch} not available. Please choose from:", available_epochs)
                continue
            break
        except ValueError:
            print("Please enter a valid number")
    
    # Get number of examples from user
    while True:
        try:
            num_examples = int(input("\nEnter the number of examples to plot (1-10): "))
            if num_examples < 1 or num_examples > 10:
                print("Please enter a number between 1 and 10")
                continue
            break
        except ValueError:
            print("Please enter a valid number")
    
    # Get batch size from user
    while True:
        try:
            batch_size = int(input("\nEnter the batch size (default is 50): ") or "50")
            if batch_size < 1:
                print("Batch size must be positive")
                continue
            break
        except ValueError:
            print("Please enter a valid number")
    
    # Load data
    _, val_loader = get_train_val_test_loaders(batch_size=batch_size)
    
    # Load generator from checkpoint
    checkpoint_path = f'checkpoints/checkpoint_epoch_{epoch}.pt'
    try:
        generator, stats = load_generator_from_checkpoint(checkpoint_path)
        print(f"\nLoaded generator from epoch {epoch}")
        
        '''
        # Plot training history
        utils.make_training_plot(f"Generator Training History (up to epoch {epoch})")
        utils.plot_training_stats(stats)
        utils.save_plot(save_path=f"images/training_history_epoch_{epoch}.png")
        '''
        
        # Evaluate generator
        print("\nEvaluating generator...")
        val_loss, recon_loss = evaluate_generator(
            generator=generator,
            val_loader=val_loader,
            num_examples=num_examples
        )
        
        print("\nResults saved in 'images' directory")
        
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found for epoch {epoch}")
        return

if __name__ == "__main__":
    main()