import torch
from pytorch_forecasting.metrics import QuantileLoss


class HorizonWeightedQuantileLoss(QuantileLoss):
    """
    Quantile (pinball) loss with horizon weighting w(h) = 1 / h^gamma.

    Subclassing QuantileLoss is load-bearing: TemporalFusionTransformer.from_dataset
    deduces the output head size via isinstance(loss, QuantileLoss), and the median
    extraction in to_prediction() relies on the QuantileLoss implementation.
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
        Calculate horizon-weighted pinball loss.

        Args:
            y_pred: network output (batch_size, n_timesteps, n_quantiles)
            target: actual values (batch_size, n_timesteps)

        Returns:
            torch.Tensor: per-element weighted losses (batch_size, n_timesteps, n_quantiles)
        """
        losses = super().loss(y_pred, target)  # (batch_size, n_timesteps, n_quantiles)

        n_timesteps = losses.shape[1]
        h = torch.arange(1, n_timesteps + 1, device=losses.device, dtype=losses.dtype)
        weights = (h ** -self.gamma).view(1, n_timesteps, 1)

        return losses * weights
