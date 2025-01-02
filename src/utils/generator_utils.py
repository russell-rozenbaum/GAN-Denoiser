'''
Generator utility functions for training and evaluating
'''
import numpy as np
import torch
from torch.nn.functional import softmax
from sklearn import metrics
from . import utils
from . import model_common_utils


def _get_metrics(mixed_loader, generator, discriminator, criterion):
    true, pred, score = [], [], []
    correct, total = 0, 0
    running_loss = []
    for i, mixed_data in enumerate(mixed_loader):
        with torch.no_grad():
            generator_out = generator.forward(mixed_data)

            discriminator_out = discriminator.forward(generator_out)
            predicted = model_common_utils.predictions(discriminator_out.data)
            true_labels = torch.ones_like(predicted)
            #print("Discriminator Output for generator:", discriminator_out)

            true.append(true_labels)
            pred.append(predicted)
            score.append(discriminator_out.data.view(-1)) # No AUROC, need?

            total += mixed_data.size(0)
            correct += (predicted == true_labels).sum().item()
            running_loss.append(
                criterion(discriminator_out, true_labels)
                .item()
            )

    true = torch.cat(true)
    pred = torch.cat(pred)
    score = torch.cat(score) # No AUROC, need?
    loss = np.mean(running_loss)
    acc = correct / total
    auroc = -1
    return loss, acc

def evaluate_epoch(
    generator,
    discriminator,
    tr_mixed_loader, 
    val_mixed_loader, 
    criterion,
    epoch,
    stats,
    axes
):
    """
    Evaluate the generator on train and validation sets, updating stats with the performance
    """
    train_loss, train_acc = _get_metrics(tr_mixed_loader, generator, discriminator, criterion)
    val_loss, val_acc = _get_metrics(val_mixed_loader, generator, discriminator, criterion)
  
    stats_at_epoch = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }

    stats.append(stats_at_epoch)
    
    utils.log_training(epoch, stats, "generator")
    utils.update_training_plot(axes, stats, "generator")

def train_epoch(generator, discriminator, mixed_data_loader, criterion, optimizer):
    """
    Train a discriminator model for one epoch using data from clean_data_loader and 
    outputs from generator given mixed_data_loader input

    Args:
        
    
    Description:
        
    Returns: None
    """
    generator.train()
    discriminator.eval()

    for i, mixed_data in enumerate(mixed_data_loader):
        # Reset optimizer gradient calculations
        optimizer.zero_grad()
        # Get generator denoised signal (forward pass)
        generator_out = generator.forward(mixed_data)
        # Get prediction for discriminator
        discriminator_out = discriminator.forward(generator_out)
        # Calculate loss between model prediction and true labels
        loss = criterion(discriminator_out, torch.ones_like(discriminator_out))
        # Perform backward pass
        loss.backward()
        # Update model weights
        optimizer.step()