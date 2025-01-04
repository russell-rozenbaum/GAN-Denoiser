'''
Discriminator utility functions for training and evaluating
'''
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from . import utils
from . import model_common_utils

def _discriminator_loss(disc_denoised_outputs, disc_clean_outputs):
    '''
    Standard GAN discriminator loss: maximize log(D(x)) + log(1 - D(G(z)))
    
    Args:
        disc_clean_outputs: Discriminator outputs on real (clean) data, shape (N, 1)
        disc_denoised_outputs: Discriminator outputs on fake (denoised/generated) data, shape (N, 1)
        delta: Scaling factor for loss
    '''
    # Avoid log(0) by adding a small epsilon for numerical stability
    epsilon = 1e-8
    
    # For real samples: maximize log(D(x))
    clean_loss = -torch.mean(torch.log(disc_clean_outputs + epsilon))
    
    # For fake samples: maximize log(1 - D(G(z)))
    denoised_loss = -torch.mean(torch.log(1 - disc_denoised_outputs + epsilon))
    
    # Total loss (negative since we want to maximize)
    total_loss = (clean_loss + denoised_loss)
    
    return total_loss

def _get_metrics(loader, generator, discriminator):
        
        true_labels, scores = [], []
        correct, total = 0, 0
        running_loss = []

        # Discriminator evaluation on denoised data
        for mixed_data, clean_data, _, _ in loader:
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
                disc_loss = _discriminator_loss(disc_denoised_out, disc_clean_out)
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
    tr_loader,
    val_loader,
    epoch,
    stats,
    axes
):
    """
    Evaluate the generator on train and validation sets, updating stats with the performance
    """

    train_loss, train_acc, train_auroc = _get_metrics(tr_loader, generator, discriminator)
    val_loss, val_acc, val_auroc = _get_metrics(val_loader, generator, discriminator)

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

def train_epoch(discriminator, generator, tr_loader, optimizer):
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

    for mixed_data, clean_data, _, _ in tr_loader:
        # Reset optimizer gradient calculations
        optimizer.zero_grad()

        # Get generator output (denoised signal)
        generator_out = generator.forward(mixed_data)

        # Get discriminator prediction on denoised data
        discriminator_denoised_out = discriminator.forward(generator_out)

        # Get discriminator prediction on clean data
        discriminator_clean_out = discriminator.forward(clean_data)

        # Calculate loss between model prediction and true labels (1 for clean)
        disc_loss = _discriminator_loss(discriminator_denoised_out, discriminator_clean_out)

        disc_loss *= discriminator.delta

        # Perform backward pass and optimizer step
        disc_loss.backward()
        optimizer.step()