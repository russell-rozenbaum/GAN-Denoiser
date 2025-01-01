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
     val_clean_loader, 
     val_mixed_loader) = get_train_val_test_loaders(batch_size=32)

    # Models
    generator = DenoisingAE()
    discriminator = Discriminator()

    criterion = torch.nn.BCELoss()
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3)

    print("Number of float-valued parameters in DenoisingAE:", train_utils.count_parameters(generator))
    print("Number of float-valued parameters in Discriminator:", train_utils.count_parameters(discriminator))

    axes = utils.make_training_plot()
    start_epoch = 0
    generator_stats = []
    discriminator_stats = []

    # Train generator for 1 epoch
    train_utils.evaluate_epoch(
        axes, 
        tr_clean_loader, 
        tr_mixed_loader, 
        val_clean_loader, 
        val_mixed_loader, 
        generator,
        discriminator,
        "generator",
        criterion, 
        start_epoch,
        generator_stats
    )

    # Train discriminator for 1 epoch
    train_utils.evaluate_epoch(
        axes, 
        tr_clean_loader, 
        tr_mixed_loader, 
        val_clean_loader, 
        val_mixed_loader, 
        generator,
        discriminator,
        "discriminator",
        criterion, 
        start_epoch, 
        discriminator_stats
    )

    prev_gen_val_loss = generator_stats[start_epoch]["val_loss"]
    prev_disc_val_loss = discriminator_stats[start_epoch]["val_loss"]

    patience = 5
    curr_count_to_patience = 0

    # Loop over the entire dataset multiple times
    epoch = start_epoch
    while curr_count_to_patience < patience:
        # Train generator
        train_utils.generator_train_epoch(
            tr_mixed_loader,
            criterion, 
            gen_optimizer,
            generator=generator,
            discriminator=discriminator,
        )

        # Evaluate generator
        train_utils.evaluate_epoch(
            axes, 
            tr_clean_loader, 
            tr_mixed_loader, 
            val_clean_loader, 
            val_mixed_loader, 
            generator, 
            discriminator,
            "generator",
            criterion, 
            epoch + 1, 
            generator_stats
        )

        # update early stopping parameters
        curr_count_to_patience, prev_gen_val_loss = train_utils.early_stopping(
            generator_stats, curr_count_to_patience, prev_gen_val_loss
        )

        # Train discriminator
        train_utils.discriminator_train_epoch(
            tr_mixed_loader, 
            tr_clean_loader, 
            criterion, 
            gen_optimizer,
            discriminator=discriminator,
            generator=generator,
        )

        # Evaluate discriminator
        train_utils.evaluate_epoch(
            axes, 
            tr_clean_loader, 
            tr_mixed_loader, 
            val_clean_loader, 
            val_mixed_loader, 
            generator, 
            discriminator,
            "discriminator",
            criterion,
            epoch + 1, 
            generator_stats
        )

        # update early stopping parameters
        curr_count_to_patience, prev_disc_val_loss = train_utils.early_stopping(
            discriminator_stats, curr_count_to_patience, prev_disc_val_loss
        )

        epoch += 1

    print("Finished Training")
    # Save figure and keep plot open
    utils.save_cnn_training_plot(patience)
    utils.hold_training_plot()




if __name__ == "__main__":
    main()
