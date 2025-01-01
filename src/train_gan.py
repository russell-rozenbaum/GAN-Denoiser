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
     val_mixed_loader) = get_train_val_test_loaders(batch_size=64)

    # Models
    generator = DenoisingAE(num_enc_filters=8, num_dec_filters=16, num_downsamples=3, num_upsamples=3)
    discriminator = Discriminator(num_filters=8)

    criterion = torch.nn.BCELoss()
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    print("Number of float-valued parameters in DenoisingAE:", train_utils.count_parameters(generator))
    print("Number of float-valued parameters in Discriminator:", train_utils.count_parameters(discriminator))

    axes = utils.make_training_plot()
    start_epoch = 0
    generator_stats = []
    discriminator_stats = []

    # Init generator stats
    train_utils.evaluate_epoch(
        axes=axes, 
        mixed_data_tr_loader=tr_mixed_loader, 
        clean_data_tr_loader=tr_clean_loader,
        mixed_data_val_loader=val_mixed_loader, 
        clean_data_val_loader=val_clean_loader,
        generator=generator,
        discriminator=discriminator,
        model_to_eval="generator",
        criterion=criterion,
        epoch=start_epoch,
        stats=generator_stats,
    )

    # Init discriminator stats
    train_utils.evaluate_epoch(
        axes=axes, 
        mixed_data_tr_loader=tr_mixed_loader, 
        clean_data_tr_loader=tr_clean_loader,
        mixed_data_val_loader=val_mixed_loader, 
        clean_data_val_loader=val_clean_loader,
        generator=generator,
        discriminator=discriminator,
        model_to_eval="discriminator",
        criterion=criterion, 
        epoch=start_epoch,
        stats=discriminator_stats,
    )

    prev_gen_val_loss = generator_stats[start_epoch]["val_loss"]
    prev_disc_val_loss = discriminator_stats[start_epoch]["val_loss"]

    patience = 50
    curr_count_to_patience = 0

    # Loop over the entire dataset multiple times
    epoch = start_epoch
    while curr_count_to_patience < patience and epoch < 80:
        # Train generator
        train_utils.generator_train_epoch(
            mixed_data_loader=tr_mixed_loader,
            criterion=criterion, 
            optimizer=gen_optimizer,
            generator=generator,
            discriminator=discriminator,
        )

        # Evaluate generator
        train_utils.evaluate_epoch(
            axes=axes, 
            mixed_data_tr_loader=tr_mixed_loader, 
            clean_data_tr_loader=tr_clean_loader,
            mixed_data_val_loader=val_mixed_loader, 
            clean_data_val_loader=val_clean_loader,
            generator=generator,
            discriminator=discriminator,
            model_to_eval="generator",
            criterion=criterion, 
            epoch=epoch + 1,
            stats=generator_stats,
        )

        # update early stopping parameters
        curr_count_to_patience, prev_gen_val_loss = train_utils.early_stopping(
            generator_stats, curr_count_to_patience, prev_gen_val_loss, patience
        )

        # Train discriminator
        train_utils.discriminator_train_epoch(
            mixed_data_loader=tr_mixed_loader,
            clean_data_loader=tr_clean_loader, 
            criterion=criterion, 
            optimizer=disc_optimizer,
            discriminator=discriminator,
            generator=generator,
        )

        # Evaluate discriminator
        train_utils.evaluate_epoch(
            axes=axes, 
            mixed_data_tr_loader=tr_mixed_loader, 
            clean_data_tr_loader=tr_clean_loader,
            mixed_data_val_loader=val_mixed_loader, 
            clean_data_val_loader=val_clean_loader,
            generator=generator,
            discriminator=discriminator,
            model_to_eval="discriminator",
            criterion=criterion, 
            epoch=epoch + 1,
            stats=discriminator_stats,
        )

        # update early stopping parameters
        curr_count_to_patience, prev_disc_val_loss = train_utils.early_stopping(
            discriminator_stats, curr_count_to_patience, prev_disc_val_loss, patience
        )

        epoch += 1

    print("Finished Training")

    utils.close_plot()

    time = np.linspace(0, 1, int(1024 * 1), endpoint=False)
    batch = next(iter(val_mixed_loader))
    random_idx = random.randint(0, batch.shape[0] - 1)
    random_mixed_signal = batch[random_idx:random_idx+1]  # Keep as [1, channel, signal_length]
    name = f"Random Mixed Signal from validation set"
    utils.plot_signal(time, random_mixed_signal.squeeze(), name)

    signal = generator.forward(random_mixed_signal)
    signal = signal.detach().cpu().numpy().squeeze()  # Detach from graph, convert to numpy, and remove extra dimensions

    name = f"Random Denoised Mixed Signal from trained generator"
    
    utils.plot_signal(time, signal, name)
    print(f"Displayed: plot of generator output")

    # Save figure and keep plot open
    utils.hold_training_plot()




if __name__ == "__main__":
    main()
