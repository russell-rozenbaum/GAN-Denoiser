"""
Helper file for common training functions.
"""

import numpy as np
import torch
from torch.nn.functional import softmax
from sklearn import metrics
import utils


def count_parameters(model):
    """Count number of learnable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def early_stopping(stats, curr_count_to_patience, prev_val_loss, patience):
    """
    Calculate new patience and validation loss.
    """

    epoch = len(stats) - 1
    curr_val_loss = stats[-1]["val_loss"]

    curr_count_to_patience = 0 if curr_val_loss < prev_val_loss else curr_count_to_patience + 1

    prev_val_loss = min(curr_val_loss, prev_val_loss)
    
    if curr_count_to_patience == patience:
        h_epoch = epoch - (patience // 2)
        print(f"Lowest Validation Loss: {prev_val_loss} \n At Epoch: {h_epoch}")

    return curr_count_to_patience, prev_val_loss


def evaluate_epoch(
    axes,
    mixed_data_tr_loader, 
    clean_data_tr_loader,
    mixed_data_val_loader, 
    clean_data_val_loader,
    generator,
    discriminator,
    model_to_eval,
    criterion,
    epoch,
    stats,
):
    """
    Evaluate the generator on train and validation sets, updating stats with the performance
    """

    # Discriminator objectives
    target_clean_label = 1
    target_denoised_label = 0

    # Generator objective
    target_denoised_label_for_generator = 1

    def _get_metrics(mixed_loader, clean_loader, model_to_eval):
        true, pred, score = [], [], []
        correct, total = 0, 0
        running_loss = []
        if model_to_eval == "generator" :
            for i, mixed_data in enumerate(mixed_loader):
                with torch.no_grad():
                    generator_out = generator.forward(mixed_data)

                    discriminator_out = discriminator.forward(generator_out)
                    predicted = predictions(discriminator_out.data)
                    true_labels = torch.ones_like(predicted)
                    print("Discriminator Output for generator:", discriminator_out)

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
            return loss, acc, auroc
        else : # model == "discriminator"
            # Discriminator evaluation
            for i, mixed_data in enumerate(mixed_loader):
                with torch.no_grad():
                    generator_out = generator.forward(mixed_data)
                    discriminator_out = discriminator.forward(generator_out)
                    predicted = predictions(discriminator_out.data)
                    true_labels = torch.zeros_like(predicted)
                    print("Discriminator Output for disc mixed:", discriminator_out)
                    true.append(true_labels)
                    pred.append(predicted)
                    score.append(discriminator_out.data.view(-1)) 
                    total += mixed_data.size(0)
                    correct += (predicted == true_labels).sum().item()
                    running_loss.append(
                        criterion(discriminator_out, true_labels)
                        .item()
                    )
            for i, clean_data in enumerate(clean_loader):
                with torch.no_grad():
                    discriminator_out = discriminator.forward(clean_data)
                    predicted = predictions(discriminator_out.data)
                    true_labels = torch.ones_like(predicted)
                    print("Discriminator Output for disc clean:", discriminator_out)
                    true.append(true_labels)
                    pred.append(predicted)
                    score.append(discriminator_out.data.view(-1)) 
                    total += clean_data.size(0)
                    correct += (predicted == true_labels).sum().item()
                    running_loss.append(
                        criterion(discriminator_out, true_labels)
                        .item()
                    )
            true = torch.cat(true)
            pred = torch.cat(pred)
            score = torch.cat(score) 
            loss = np.mean(running_loss)
            acc = correct / total
            auroc = metrics.roc_auc_score(true, score)
            return loss, acc, auroc

    train_loss, train_acc, train_auroc = _get_metrics(mixed_data_tr_loader, clean_data_tr_loader, model_to_eval)
    val_loss, val_acc, val_auroc = _get_metrics(mixed_data_val_loader, clean_data_val_loader, model_to_eval)
    if model_to_eval == "generator" :
        stats_at_epoch = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
    else :
        stats_at_epoch = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_auroc": train_auroc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_auroc": val_auroc,
        }
    stats.append(stats_at_epoch)
    
    utils.log_training(epoch, stats, model_to_eval)
    utils.update_training_plot(axes, epoch, stats, model_to_eval)



def discriminator_train_epoch(mixed_data_loader, clean_data_loader, criterion, optimizer, discriminator, generator):
    """
    Train a discriminator model for one epoch using data from clean_data_loader and 
    outputs from generator given mixed_data_loader input

    Args:
        
    
    Description:
        
    Returns: None
    """
    discriminator.train()
    generator.eval()

    correct_clean_label = 1
    for i, clean_data in enumerate(clean_data_loader):
        # Reset optimizer gradient calculations
        optimizer.zero_grad()
        # Get discriminator prediction
        discriminator_out = discriminator.forward(clean_data)
        # Calculate loss between model prediction and true labels
        loss = criterion(discriminator_out, torch.ones_like(discriminator_out))
        # Perform backward pass
        loss.backward()
        # Update model weights
        optimizer.step()
    
    correct_denoised_label = 0
    for i, mixed_data in enumerate(mixed_data_loader):
        # Reset optimizer gradient calculations
        optimizer.zero_grad()
        # Get generator denoised signal (forward pass)
        generator_out = generator.forward(mixed_data)
        # Get prediction for discriminator
        discriminator_out = discriminator.forward(generator_out)
        # Calculate loss between model prediction and true labels
        loss = criterion(discriminator_out, torch.zeros_like(discriminator_out))
        # Perform backward pass
        loss.backward()
        # Update model weights
        optimizer.step()

def generator_train_epoch(mixed_data_loader, criterion, optimizer, discriminator, generator):
    """
    Train a discriminator model for one epoch using data from clean_data_loader and 
    outputs from generator given mixed_data_loader input

    Args:
        
    
    Description:
        
    Returns: None
    """
    generator.train()
    discriminator.eval()

    correct_denoised_label = 0
    target_label_for_generator = 1 - correct_denoised_label
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


def predictions(logits, threshold=0.5):
    """
    Args:
        logits: A tensor of shape (batch_size, 1) 
                containing sigmoid outputs from the discriminator.

    Returns:
        a tensor rounded to {0, 1}.
    """
    return (logits >= threshold).float()