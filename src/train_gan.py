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
    tr_loader, val_loader = get_train_val_test_loaders(batch_size=8)

    # Models
    generator = DenoisingAE(num_enc_filters=12, num_dec_filters=12, num_downsamples=4, num_upsamples=4)
    discriminator = Discriminator(num_filters=4, num_convolutions=3, fc_units=8)

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

    # TODO: Tweak patience hyperparam
    patience = 50000
    # Init patience count and val_losses
    curr_count_to_patience = 0
    prev_gen_val_loss = generator_stats[start_epoch]["val_loss"]
    prev_disc_val_loss = discriminator_stats[start_epoch]["val_loss"]

    epoch = start_epoch

    # While early stopping or max_epochs not reached, alternate between training generator and discriminator
    while curr_count_to_patience < patience and epoch < 30:
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

    print("Finished Training")

    # TODO: Remove dense code below vvv --- strictly for debugging purposes
    utils.close_plot()
    # Loop to display 3 examples
    for i in range(3):  # Adjust to 3 examples
        # Get a batch of mixed and clean signals from the validation set
        batch = next(iter(val_loader))  # Assuming val_loader is an iterator
        
        mixed_data, clean_data, mixed_file, clean_file = batch  # Unpack the batch
        
        # Choose a random index to display
        random_idx = random.randint(0, mixed_data.shape[0] - 1)
        
        # Get the corresponding mixed and clean signals for this random index
        random_mixed_signal = mixed_data[random_idx:random_idx + 1]  # Shape: [1, 1, signal_length]
        random_clean_signal = clean_data[random_idx:random_idx + 1]  # Shape: [1, 1, signal_length]
        
        # Create a time array for plotting
        time = np.linspace(0, 1, int(random_mixed_signal.shape[-1]), endpoint=False)
        
        # Plot the random mixed signal
        name = f"Random Mixed Signal #{i + 1} from validation set"
        utils.plot_signal(time, random_mixed_signal.squeeze(), name)
        
        # Plot the corresponding clean (sine wave) signal
        name = f"Corresponding Clean Signal #{i + 1} (Sine Wave)"
        utils.plot_signal(time, random_clean_signal.squeeze(), name)

        # Plot the generator output
        generated_signal = generator(random_mixed_signal).detach().cpu().numpy().squeeze()
        name = f"Corresponding Denoised Signal #{i + 1}"
        utils.plot_signal(time, generated_signal, name)
        
        print(f"Displayed: mixed and clean signals #{i + 1}")
    # TODO: Remove above ^^^

    # Save figure and keep plot open
    utils.hold_training_plot()




if __name__ == "__main__":
    main()
