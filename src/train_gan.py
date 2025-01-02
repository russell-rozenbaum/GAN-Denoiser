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


def main():
    
    # Data Splits
    (tr_clean_loader,
     tr_mixed_loader, 
     val_clean_loader, 
     val_mixed_loader) = get_train_val_test_loaders(batch_size=64)

    # Models
    generator = DenoisingAE(num_enc_filters=6, num_dec_filters=4, num_downsamples=3, num_upsamples=3)
    discriminator = Discriminator(num_filters=4, num_convolutions=3, fc_units=16)

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
        tr_mixed_loader=tr_mixed_loader, 
        val_mixed_loader=val_mixed_loader, 
        epoch=start_epoch,
        stats=generator_stats,
        axes=axes,
    )

    # Init discriminator stats
    disc_utils.evaluate_epoch(
        discriminator=discriminator,
        generator=generator,
        tr_mixed_loader=tr_mixed_loader, 
        val_mixed_loader=val_mixed_loader, 
        tr_clean_loader=tr_clean_loader,
        val_clean_loader=val_clean_loader,
        epoch=start_epoch,
        stats=discriminator_stats,
        axes=axes,
    )

    # TODO: Tweak patience hyperparam
    patience = 50000
    # Init patience count and val_losses
    curr_count_to_patience = 0
    prev_gen_val_loss = generator_stats[start_epoch]["val_loss"]
    prev_disc_val_loss = discriminator_stats[start_epoch]["val_loss"]

    epoch = start_epoch

    # While early stopping or max_epochs not reached, alternate between training generator and discriminator
    while curr_count_to_patience < patience and epoch < 100:
        # Train generator
        gen_utils.train_epoch(
            generator=generator,
            discriminator=discriminator,
            mixed_data_loader=tr_mixed_loader,
            optimizer=gen_optimizer,
        )

        # Train discriminator
        disc_utils.train_epoch(
            discriminator=discriminator,
            generator=generator,
            mixed_data_loader=tr_mixed_loader,
            clean_data_loader=tr_clean_loader, 
            optimizer=disc_optimizer,
        )

        # Increment epoch, as we've performed another iterations of training
        epoch += 1

        # Init generator stats
        gen_utils.evaluate_epoch(
            generator=generator,
            discriminator=discriminator,
            tr_mixed_loader=tr_mixed_loader, 
            val_mixed_loader=val_mixed_loader, 
            epoch=epoch,
            stats=generator_stats,
            axes=axes,
        )

        # Init discriminator stats
        disc_utils.evaluate_epoch(
            discriminator=discriminator,
            generator=generator,
            tr_mixed_loader=tr_mixed_loader, 
            val_mixed_loader=val_mixed_loader, 
            tr_clean_loader=tr_clean_loader,
            val_clean_loader=val_clean_loader,
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

    print("Finished Training")

    # TODO: Remove dense code below vvv --- strictly for debugging purposes
    utils.close_plot()
    for i in range(5) :
        time = np.linspace(0, 1, int(1024 * 1), endpoint=False)
        batch = next(iter(val_mixed_loader))
        random_idx = random.randint(0, batch.shape[0] - 1)
        random_mixed_signal = batch[random_idx:random_idx+1]  # Keep as [1, channel, signal_length]
        name = f"Random Mixed Signal #{i + 1} from validation set"
        utils.plot_signal(time, random_mixed_signal.squeeze(), name)
        signal = generator.forward(random_mixed_signal)
        signal = signal.detach().cpu().numpy().squeeze()  # Detach from graph, convert to numpy, and remove extra dimensions
        name = f"Random Denoised Mixed Signal #{i + 1} from trained generator"
        #signal = utils.normalize_signal(signal)
        utils.plot_signal(time, signal, name)
        print(f"Displayed: plot of generator output")
    # TODO: Remove above ^^^

    # Save figure and keep plot open
    utils.hold_training_plot()




if __name__ == "__main__":
    main()
