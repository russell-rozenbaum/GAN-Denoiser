"""
Train GAN
    Trains a GAN which has a DenoisingAE and Discriminator Network
"""
import torch
import numpy as np
import random
from dataset import get_train_val_test_loaders
from models.discriminator import Discriminator
from models.generator import DenoisingAE
import utils.utils as utils
import utils.generator_utils as gen_utils
import utils.discriminator_utils as disc_utils
import utils.model_common_utils as model_utils
from torch.optim.lr_scheduler import StepLR

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def train_gan(
    # Generator params
    num_enc_filters=6,
    num_dec_filters=12,
    num_upsamples_and_downsamples=3,
    # Discriminator params
    num_filters=2,
    num_convolutions=3,
    num_fc_units=8,
    # Training hyperparams
    max_epochs=100,
    patience=20,
    batch_size=16,
):

    # Data Splits
    tr_loader, val_loader = get_train_val_test_loaders(batch_size=batch_size)

    # Models
    generator = DenoisingAE(
        num_enc_filters=num_enc_filters, 
        num_dec_filters=num_dec_filters, 
        num_downsamples=num_upsamples_and_downsamples, 
        num_upsamples=num_upsamples_and_downsamples
    )

    discriminator = Discriminator(
        num_filters=num_filters,
        num_convolutions=num_convolutions, 
        fc_units=num_fc_units
    )

    # Hyperparams
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    # TODO: Activate schedulers for decreasing learning rate
    gen_scheduler = StepLR(gen_optimizer, step_size=40, gamma=np.sqrt(10)/10)
    disc_scheduler = StepLR(disc_optimizer, step_size=40, gamma=np.sqrt(10)/10)

    print("Number of float-valued parameters in DenoisingAE:", model_utils.count_parameters(generator))
    print("Number of float-valued parameters in Discriminator:", model_utils.count_parameters(discriminator))

    # Make training plot for generator and discriminator stats
    axes = utils.make_training_plot()
    start_epoch = 0
    generator_stats = []
    discriminator_stats = []

    # Init generator stats
    gen_utils.evaluate_epoch(
        generator=generator,
        discriminator=discriminator,
        tr_loader=tr_loader,
        val_loader=val_loader,
        epoch=start_epoch,
        stats=generator_stats,
        axes=axes,
    )

    # Init discriminator stats
    disc_utils.evaluate_epoch(
        discriminator=discriminator,
        generator=generator,
        tr_loader=tr_loader,
        val_loader=val_loader,
        epoch=start_epoch,
        stats=discriminator_stats,
        axes=axes,
    )

    # Init patience count and val_losses
    curr_count_to_patience = 0
    prev_gen_val_loss = generator_stats[start_epoch]["val_loss"]
    prev_disc_val_loss = discriminator_stats[start_epoch]["val_loss"]

    epoch = start_epoch

    # While early stopping or max_epochs not reached, alternate between training generator and discriminator
    while curr_count_to_patience < patience and epoch < max_epochs:
        # Train generator
        gen_utils.train_epoch(
            generator=generator,
            discriminator=discriminator,
            tr_loader=tr_loader,
            optimizer=gen_optimizer,
        )

        # Train discriminator
        disc_utils.train_epoch(
            discriminator=discriminator,
            generator=generator,
            tr_loader=tr_loader,
            optimizer=disc_optimizer,
        )

        # Increment epoch, as we've performed another iterations of training
        epoch += 1

        # Init generator stats
        gen_utils.evaluate_epoch(
            generator=generator,
            discriminator=discriminator,
            tr_loader=tr_loader,
            val_loader=val_loader,
            epoch=epoch,
            stats=generator_stats,
            axes=axes,
        )

        # Init discriminator stats
        disc_utils.evaluate_epoch(
            discriminator=discriminator,
            generator=generator,
            tr_loader=tr_loader,
            val_loader=val_loader,
            epoch=epoch,
            stats=discriminator_stats,
            axes=axes,
        )

        # Update generator early stopping parameters
        curr_count_to_patience, prev_gen_val_loss = model_utils.early_stopping(
            generator_stats, curr_count_to_patience, prev_gen_val_loss, patience, epoch
        )

        # Update discriminator early stopping parameters
        curr_count_to_patience, prev_disc_val_loss = model_utils.early_stopping(
            discriminator_stats, curr_count_to_patience, prev_disc_val_loss, patience, epoch
        )

        gen_scheduler.step()
        disc_scheduler.step()

    # Training finsihed
    print("Max epochs reached") if epoch >= max_epochs else print("Patience reached")
    print("Finished Training")

    # Save the performance training plot
    utils.save_plot(save_path="images/performance_plot.png")
    utils.hold_plot()

    # Plot 2 examples of the generator outputs and save them
    title = f"Clean/Mixed/Denoised Signals\nGenerator Params: num_enc_filters={num_enc_filters}, num_dec_filters={num_dec_filters}, num_down/upsamples={num_upsamples_and_downsamples}\nDiscriminator Params: num_filters={num_filters}, convolutional_layers={num_convolutions}, fc_units={num_fc_units}\nepochs={epoch}, max_epochs={max_epochs}, patience={patience}"
    utils.plot_denoising_results(generator=generator, val_loader=val_loader, num_examples=2, title=title)
    utils.save_plot(save_path="images/denoised_examples.png")
    utils.hold_plot()


def main() :

    train_gan(
        # Generator params
        num_enc_filters=6,
        num_dec_filters=12,
        num_upsamples_and_downsamples=3,
        # Discriminator params
        num_filters=2,
        num_convolutions=3,
        num_fc_units=8,
        # Training hyperparams
        max_epochs=100,
        patience=20,
        batch_size=16,
    )


if __name__ == "__main__":
    main()
