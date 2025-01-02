'''
Generator utility functions for training and evaluating
'''
import numpy as np
import torch
from torch.nn.functional import softmax
from sklearn import metrics
from . import utils
from . import model_common_utils

REAL_LABEL=1
FAKE_LABEL=0

def _adversarial_loss(disc_outputs):
    '''
    Calculate MSE Loss

    Description:
        Simple adversarial loss calculation. How well was generator able to fool
        discriminator into believing it is clean data?
    '''
    gen_losses = (1 - disc_outputs) ** 2  # Tensor of shape (batch_size,)
    mean_loss = torch.mean(gen_losses)  # Scalar tensor
    print(f"mean loss as: {mean_loss}")
    return mean_loss, gen_losses

def _reconstruction_loss(gen_outputs, clean_signals):
    '''
    Calculate L1 Norm between generator output and true clean signal

    Description:
        AE Loss style -- generator(mixed_signal) should be as close to
        true clean signal as possible
    '''
    loss = 0
    gen_losses = []
    for denoised_data in gen_outputs:
        l = abs(denoised_data - clean_signals)
        gen_losses.append(l)
        loss += l
    
    return loss, gen_losses

def _generator_loss(gen_outputs, disc_outputs, clean_signals):
    '''
    Calculate total losses for generator

    Desciption:
        Uses both adversarial loss from GAN and reconstruction loss in basic AE style
    '''
    adv_loss, adv_gen_losses = _adversarial_loss(disc_outputs)
    # TODO: Implement passing of clean data
    recon_loss, recon_gen_losses = 0, 0 #_reconstruction_loss(gen_outputs, clean_signals)

    return adv_loss, adv_gen_losses

def _get_metrics(mixed_loader, generator, discriminator):
    correct, total = 0, 0
    running_loss = []
    for i, mixed_data in enumerate(mixed_loader):
        with torch.no_grad():
            # Get model outputs
            generator_out = generator.forward(mixed_data)
            discriminator_out = discriminator.forward(generator_out)

            # Accumulate accuracy metrics
            predicted = model_common_utils.predictions(discriminator_out.data)
            true_labels = torch.ones_like(predicted)
            correct += (predicted == true_labels).sum().item()
            total += mixed_data.size(0)

            # Accumulate loss metrics
            gen_loss, _ = _generator_loss(generator_out, discriminator_out, [])
            running_loss.append(gen_loss)

    loss = np.mean(running_loss)
    acc = correct / total
    print("Loss and acc: ")
    print(loss, acc)
    return loss, acc

def evaluate_epoch(
    generator,
    discriminator,
    tr_mixed_loader, 
    val_mixed_loader, 
    epoch,
    stats,
    axes
):
    """
    Evaluate the generator on train and validation sets, updating stats with the performance
    """
    train_loss, train_acc = _get_metrics(tr_mixed_loader, generator, discriminator)
    val_loss, val_acc = _get_metrics(val_mixed_loader, generator, discriminator)
  
    stats_at_epoch = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }

    stats.append(stats_at_epoch)
    
    utils.log_training(epoch, stats, "generator")
    utils.update_training_plot(axes, stats, "generator")

def train_epoch(generator, discriminator, mixed_data_loader, optimizer):
    """
    Train a discriminator model for one epoch using data from clean_data_loader and 
    outputs from generator given mixed_data_loader input

    Args:
        
    
    Description:
        
    Returns: None
    """
    generator.train()
    discriminator.eval()

    total_gen_loss = 0.0
    num_batches = len(mixed_data_loader)

    for i, mixed_data in enumerate(mixed_data_loader):
        # Reset optimizer gradients
        optimizer.zero_grad()

        # Forward pass through the generator
        gen_outputs = generator(mixed_data)

        # Get discriminator predictions for the generated data
        disc_outputs = discriminator(gen_outputs)

        # Calculate generator loss using the helper function
        loss, _ = _generator_loss(gen_outputs, disc_outputs, [])

        # Backward pass and optimizer step
        loss.backward()
        optimizer.step()

        # Accumulate loss for monitoring
        total_gen_loss += loss.item()

    # Return average generator loss
    return total_gen_loss / num_batches