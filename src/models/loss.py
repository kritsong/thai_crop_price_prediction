import torch
from pytorch_forecasting.metrics import MultiHorizonMetric
import torch.nn.functional as F

class HorizonWeightedQuantileLoss(MultiHorizonMetric):
    """
    Quantile loss with horizon weighting w(h) = 1 / h^gamma.
    """
    def __init__(
        self,
        quantiles=[0.1, 0.5, 0.9],
        gamma=2.0,
        **kwargs
    ):
        super().__init__(quantiles=quantiles, **kwargs)
        self.gamma = gamma

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate loss.

        Args:
            y_pred: network output (batch_size, n_timesteps, n_quantiles)
            target: actual values (batch_size, n_timesteps)

        Returns:
            torch.Tensor: loss
        """
        # (batch_size, n_timesteps, n_quantiles)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - y_pred[..., i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
            
        losses = torch.cat(losses, dim=-1) # (batch_size, n_timesteps, n_quantiles)
        
        # apply horizon weights
        n_timesteps = losses.shape[1]
        h = torch.arange(1, n_timesteps + 1, device=losses.device, dtype=losses.dtype)
        weights = 1.0 / (h ** self.gamma) # (n_timesteps,)
        
        # expand weights to match shape
        weights = weights.view(1, n_timesteps, 1)
        
        weighted_losses = losses * weights
        
        return weighted_losses
