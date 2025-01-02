'''
Shared model utilty functions
'''

def count_parameters(model):
    """
    Count number of learnable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def predictions(logits, threshold=0.5):
    """
    Args:
        logits: A tensor of shape (batch_size, 1) 
                containing sigmoid outputs from the discriminator.

    Returns:
        a tensor rounded to {0, 1}.
    """
    return (logits >= threshold).float()

def early_stopping(stats, curr_count_to_patience, prev_val_loss, patience, epoch):
    """
    Calculate new patience and validation loss.
    """
    curr_val_loss = stats[epoch]["val_loss"]
    curr_count_to_patience = 0 if curr_val_loss < prev_val_loss else curr_count_to_patience + 1
    prev_val_loss = min(curr_val_loss, prev_val_loss)
    if curr_count_to_patience == patience:
        h_epoch = epoch - patience
        print("Patience reached")
        print(f"Lowest Validation Loss: {prev_val_loss} \n At Epoch: {h_epoch}")
    return curr_count_to_patience, prev_val_loss