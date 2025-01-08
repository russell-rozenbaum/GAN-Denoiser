'''
Generator utility functions for training and evaluating
'''
import numpy as np
import torch
import torch.nn.functional as F
from . import utils
from . import model_common_utils

REAL_LABEL=1
FAKE_LABEL=0

def _adversarial_loss(disc_outputs, gamma):
    '''
    Generator adversarial loss: minimize log(1 - D(G(z))) 
    which is equivalent to maximizing log(D(G(z)))
    
    Args:
        disc_outputs: Discriminator outputs on generated data, shape (N, 1)
        gamma: Scaling factor for loss
    '''
    epsilon = 1e-8
    
    # Maximize log(D(G(z))) - this tricks discriminator better than minimizing log(1 - D(G(z)))
    gen_loss = -torch.mean(torch.log(disc_outputs + epsilon))
    
    return gen_loss * gamma

def _reconstruction_loss(gen_outputs, clean_signals, rho):
    '''
    Calculate L1 Norm between generator output and true clean signal

    Description:
        AE Loss style -- generator(mixed_signal) should be as close to
        true clean signal as possible
    '''
    
    # Calculate the absolute difference (L1 norm) between the generated and clean signals
    gen_losses = torch.abs(gen_outputs - clean_signals)  # Tensor of shape (batch_size, signal_length)
    
    # Mean L1 loss across the batch
    mean_loss = torch.mean(gen_losses)  # Scalar tensor representing the mean L1 loss
    
    return mean_loss * rho

def _generator_loss(gen_outputs, disc_outputs, clean_signals, gamma, rho):
    '''
    Calculate total losses for generator

    Desciption:
        Uses both adversarial loss from GAN and reconstruction loss in basic AE style
    '''
    adv_loss = _adversarial_loss(disc_outputs, gamma)
    recon_loss = _reconstruction_loss(gen_outputs, clean_signals, rho)

    total_gen_loss = adv_loss + recon_loss

    return total_gen_loss

def _get_metrics(loader, generator, discriminator):
    correct, total = 0, 0
    running_loss = []
    for mixed_data, clean_data, _, _ in loader :
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
            gen_loss = _generator_loss(generator_out, discriminator_out, clean_data, generator.gamma, generator.rho)
            running_loss.append(gen_loss)

    loss = np.mean(running_loss)
    acc = correct / total
    print("Loss and acc: ")
    print(loss, acc)
    return loss, acc

def evaluate_epoch(
    generator,
    discriminator,
    tr_loader,
    val_loader,
    epoch,
    stats,
    axes
):
    """
    Evaluate the generator on train and validation sets, updating stats with the performance
    """
    train_loss, train_acc = _get_metrics(tr_loader, generator, discriminator)
    val_loss, val_acc = _get_metrics(val_loader, generator, discriminator)
  
    stats_at_epoch = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }

    stats.append(stats_at_epoch)
    
    utils.log_training(epoch, stats, "generator")
    utils.update_training_plot(axes, stats, "generator")

def train_epoch(generator, discriminator, tr_loader, optimizer):
    """
    Train a discriminator model for one epoch using data from clean_data_loader and 
    outputs from generator given mixed_data_loader input

    Args:
        
    
    Description:
        
    Returns: None
    """
    generator.train()
    discriminator.eval()

    for mixed_data, clean_data, mixed_file, clean_file in tr_loader:
        # Reset optimizer gradients
        optimizer.zero_grad()

        #print(f"Processing Mixed Data File: {mixed_file}, Clean Data File: {clean_file}")

        # Forward pass through the generator
        gen_outputs = generator(mixed_data)

        # Get discriminator predictions for the generated data
        disc_outputs = discriminator(gen_outputs)

        # Calculate generator loss using the helper function
        loss = _generator_loss(gen_outputs, disc_outputs, clean_data, generator.gamma, generator.rho)

        # Backward pass and optimizer step
        loss.backward()
        optimizer.step()