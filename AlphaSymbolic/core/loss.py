
import torch
import torch.nn as nn

class QuantileLoss(nn.Module):
    """
    Quantile Loss (Pinball Loss) for multiple quantiles.
    
    Args:
        quantiles (list): List of quantiles to estimate (e.g. [0.25, 0.5, 0.75])
    """
    def __init__(self, quantiles=[0.25, 0.5, 0.75]):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, preds, target):
        """
        preds: [batch, num_quantiles] - Predicted values for each quantile
        target: [batch, 1] - True scalar target
        """
        # Ensure target matches batch dim
        # target shape might be [batch] or [batch, 1]
        if target.dim() == 1:
            target = target.unsqueeze(1)
            
        loss = 0
        for i, q in enumerate(self.quantiles):
            error = target - preds[:, i:i+1]
            # Pinball loss: max(q * error, (q - 1) * error)
            # Equivalent to: error * (q - I(error < 0))
            loss += torch.max(q * error, (q - 1) * error).mean()
            
        return loss
