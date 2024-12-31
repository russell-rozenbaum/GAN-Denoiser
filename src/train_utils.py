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


def early_stopping(stats, curr_count_to_patience, prev_val_loss):
    """
    Calculate new patience and validation loss.
    """

    epoch = len(stats) - 1 
    curr_val_loss = stats[-1][1]

    curr_count_to_patience = 0 if curr_val_loss < prev_val_loss else curr_count_to_patience + 1

    prev_val_loss = min(curr_val_loss, prev_val_loss)
    
    if curr_count_to_patience==5:
        h_epoch = epoch - 5
        print(f"Lowest Validation Loss: {prev_val_loss} \n At Epoch: {h_epoch}")

    return curr_count_to_patience, prev_val_loss


def evaluate_epoch(
    axes,
    tr_loader,
    val_loader,
    te_loader,
    model,
    criterion,
    epoch,
    stats,
    include_test=False,
    update_plot=True,
    multiclass=False,
):
    """
    Evaluate the `model` on the train, validation, and optionally test sets on the specified 'criterion' at the given 'epoch'.
    """
    model.eval()
    def _get_metrics(loader):
        '''
            Evaluates the model on the given loader (either train, val, or test) and returns the accuracy, loss, and AUC.
        '''
        y_true, y_pred, y_score = [], [], []
        correct, total = 0, 0
        running_loss = []
        for X, y in loader:
            with torch.no_grad():
                output = model(X)
                predicted = predictions(output.data)
                y_true.append(y)
                y_pred.append(predicted)
                if not multiclass:
                    y_score.append(softmax(output.data, dim=1)[:, 1])
                else:
                    y_score.append(softmax(output.data, dim=1))
                total += y.size(0)
                correct += (predicted == y).sum().item()
                running_loss.append(criterion(output, y).item())
        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
        y_score = torch.cat(y_score)
        loss = np.mean(running_loss)
        acc = correct / total
        if not multiclass:
            auroc = metrics.roc_auc_score(y_true, y_score)
        else:
            auroc = metrics.roc_auc_score(y_true, y_score, multi_class="ovo")
        return acc, loss, auroc

    train_acc, train_loss, train_auc = _get_metrics(tr_loader)
    val_acc, val_loss, val_auc = _get_metrics(val_loader)

    stats_at_epoch = [
        val_acc,
        val_loss,
        val_auc,
        train_acc,
        train_loss,
        train_auc,
    ]
    if include_test:
        stats_at_epoch += list(_get_metrics(te_loader))

    stats.append(stats_at_epoch)
    utils.log_training(epoch, stats)
    if update_plot:
        utils.update_training_plot(axes, epoch, stats)


def train_epoch(data_loader, model, criterion, optimizer):
    """
    Train the `model` for one epoch using data from `data_loader`.

    Args:
        data_loader: DataLoader providing batches of input data and corresponding labels.

        model: The model to be trained. This is one of the model classes in the 'model' folder. 

        criterion (torch.nn.Module): The loss function used to compute the model's loss.

        optimizer: The optimizer used to update the model parameters.

    Description:
        This function sets the model to training mode and use the data loader to iterate through the entire dataset.
        For each batch, it performs the following steps:
        1. Resets the gradient calculations in the optimizer.
        2. Performs a forward pass to get the model predictions.
        3. Computes the loss between predictions and true labels using the specified `criterion`.
        4. Performs a backward pass to calculate gradients.
        5. Updates the model weights using the `optimizer`.
    
    Returns: None
    """
    model.train()
    for i, (X, y) in enumerate(data_loader):
        # training steps

        # Reset optimizer gradient calculations
        optimizer.zero_grad()

        # Get model predictions (forward pass)
        y_pred = model.forward(X)

        # Calculate loss between model prediction and true labels
        loss = criterion(y_pred, y)

        # Perform backward pass
        loss.backward()

        # Update model weights
        optimizer.step()



def predictions(logits):
    """Determine predicted class index given logits.

    args: 
        logits (torch.Tensor): The model's output logits. It is a 2D tensor of shape (batch_size, num_classes). 

    Returns:
        the predicted class output that has the highest probability as a PyTorch Tensor. This should be of size (batch_size,).
    """
    preds = torch.zeros(logits.shape[0])

    for idx in range(logits.shape[0]) :
        max_class = -np.inf
        max_idx = -1
        for i, logit in enumerate(logits[idx]) :
            max_class = max(max_class, logit)
            max_idx = i if max_class == logit else max_idx
        preds[idx] = max_idx

    return preds

