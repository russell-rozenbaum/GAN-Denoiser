'''
Discriminator utility functions for training and evaluating
'''
import numpy as np
import torch
from torch.nn.functional import softmax
from sklearn import metrics
from . import utils
from . import model_common_utils

def _discriminator_loss(disc_denoised_outputs, disc_clean_outputs):
    '''
    Inputs:
        - disc_clean_outputs: Outputs from discriminator on clean signal input
            Shape: (N, 1)
        - disc_denoised_outputs: Outputs from discriminator on mixed signal input
            Shape: (N, 1)
    '''
    # Calculate losses
    clean_losses = (1 - disc_clean_outputs) ** 2  # Tensor of shape (batch_size,)
    denoised_losses = disc_denoised_outputs ** 2  # Tensor of shape (batch_size,)

    # Compute mean losses
    mean_clean_loss = torch.mean(clean_losses)  # Scalar tensor
    mean_denoised_loss = torch.mean(denoised_losses)  # Scalar tensor

    # Combine mean losses
    mean_loss = (mean_clean_loss + mean_denoised_loss) / 2  # Scalar tensor

    # Return as necessary
    return mean_loss, clean_losses, denoised_losses

def _get_metrics(mixed_loader, clean_loader, generator, discriminator):
        
        true_labels, scores = [], []
        correct, total = 0, 0
        running_loss = []

        # Discriminator evaluation on denoised data
        for mixed_data, clean_data in zip(mixed_loader, clean_loader):
            '''
            mixed_data and clean_data with shape (Batch_Size, Channel_Size, Sample_Rate)
            Shape: (N, 1, 1024)
            '''
            with torch.no_grad():
                # Get model outputs
                generator_out = generator.forward(mixed_data)
                disc_denoised_out = discriminator.forward(generator_out)
                disc_clean_out = discriminator.forward(clean_data)

                # Get predictions for correct/total accuracy calculation
                denoised_true_labels = torch.zeros_like(disc_denoised_out)
                true_labels.append(denoised_true_labels)
                clean_true_labels = torch.ones_like(disc_clean_out)
                true_labels.append(clean_true_labels)

                denoised_predicted = model_common_utils.predictions(disc_denoised_out.data)
                clean_predicted = model_common_utils.predictions(disc_clean_out.data)
                correct += (denoised_predicted == denoised_true_labels).sum().item()
                correct += (clean_predicted == clean_true_labels).sum().item()
                total += (mixed_data.size(0) + clean_data.size(0))

                # Accumulate auroc calculation
                scores.append(disc_denoised_out.data.view(-1))
                scores.append(disc_clean_out.data.view(-1))

                # Accumulate losses for average loss calculation
                disc_loss, _, _ = _discriminator_loss(disc_denoised_out, disc_clean_out)
                running_loss.append(disc_loss)

        true = torch.cat(true_labels)
        score = torch.cat(scores)
        loss = np.mean(running_loss)
        acc = correct / total
        auroc = metrics.roc_auc_score(true, score)
        print("Loss and acc and auroc: ")
        print(loss, acc, auroc)
        return loss, acc, auroc


def evaluate_epoch(
    discriminator,
    generator,
    tr_mixed_loader, 
    val_mixed_loader,
    tr_clean_loader,
    val_clean_loader,
    epoch,
    stats,
    axes
):
    """
    Evaluate the generator on train and validation sets, updating stats with the performance
    """

    train_loss, train_acc, train_auroc = _get_metrics(tr_mixed_loader, tr_clean_loader, generator, discriminator)
    val_loss, val_acc, val_auroc = _get_metrics(val_mixed_loader, val_clean_loader, generator, discriminator)

    stats_at_epoch = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_auroc": train_auroc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_auroc": val_auroc,
    }
    stats.append(stats_at_epoch)
    
    utils.log_training(epoch, stats, "discriminator")
    utils.update_training_plot(axes, stats, "discriminator")

def train_epoch(discriminator, generator, mixed_data_loader, clean_data_loader, optimizer):
    """
    Train a discriminator model for one epoch using data from clean_data_loader and 
    outputs from generator given mixed_data_loader input

    Args:
        discriminator: The discriminator model
        generator: The generator model
        mixed_data_loader: DataLoader for mixed data (noisy signals)
        clean_data_loader: DataLoader for clean data
        optimizer: Optimizer for training

    Returns: None
    """
    discriminator.train()
    generator.eval()

    # Track running losses
    running_clean_losses = []
    running_denoised_losses = []

    for mixed_data, clean_data in zip(clean_data_loader, mixed_data_loader):
        # Reset optimizer gradient calculations
        optimizer.zero_grad()
        # Get generator output (denoised signal)
        generator_out = generator.forward(mixed_data)
        # Get discriminator prediction on denoised data
        discriminator_denoised_out = discriminator.forward(generator_out)
        # Get discriminator prediction on clean data
        discriminator_clean_out = discriminator.forward(clean_data)
        # Calculate loss between model prediction and true labels (1 for clean)
        disc_loss, _, _ = _discriminator_loss(discriminator_denoised_out, discriminator_clean_out)
        # Perform backward pass
        disc_loss.backward()
        # Update model weights
        optimizer.step()

