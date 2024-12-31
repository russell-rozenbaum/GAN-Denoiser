"""
Train GAN
    Trains a GAN which has a DenoisingAE and Discriminator Network
"""
import torch
import numpy as np
import random
from dataset import get_train_val_test_loaders
from discriminator import Discriminator
from generator import DenoisingAE
import utils
import train_utils

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def main():
    
    # Data Splits
    (tr_clean_loader, 
     tr_mixed_loader, 
     va_clean_loader, 
     va_mixed_loader) = get_train_val_test_loaders(batch_size=1024)

    # Models
    generator = DenoisingAE()
    discriminator = Discriminator()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Number of float-valued parameters in DenoisingAE:", train_utils.count_parameters(generator))
    print("Number of float-valued parameters in Discriminator:", train_utils.count_parameters(discriminator))

    axes = utils.make_training_plot()
    start_epoch = 0
    stats = 

    # Train generator for 1 epoch
    train_utils.evaluate_epoch(
        axes, 
        tr_clean_loader, 
        tr_mixed_loader, 
        va_clean_loader, 
        va_mixed_loader, 
        generator, 
        criterion, 
        start_epoch, 
        stats
    )

    # Train discriminator for 1 epoch
    train_utils.evaluate_epoch(
        axes, 
        tr_clean_loader, 
        tr_mixed_loader, 
        va_clean_loader, 
        va_mixed_loader, 
        discriminator, 
        criterion, 
        start_epoch, 
        stats
    )

    # initial val loss for early stopping
    prev_val_loss = stats[0][1]

    patience = 5
    curr_count_to_patience = 0

    # Loop over the entire dataset multiple times
    epoch = start_epoch
    while curr_count_to_patience < patience:
        # Train model
        train_epoch(tr_loader, model, criterion, optimizer)

        # Evaluate model
        evaluate_epoch(
            axes, tr_loader, va_loader, te_loader, model, criterion, epoch + 1, stats, include_test=True
        )

        # Save model parameters
        save_checkpoint(model, epoch + 1, config("target.checkpoint"), stats)

        # update early stopping parameters
        curr_count_to_patience, prev_val_loss = early_stopping(
            stats, curr_count_to_patience, prev_val_loss
        )





if __name__ == "__main__":
    main()
