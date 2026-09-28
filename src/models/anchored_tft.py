"""Temporal Fusion Transformer that forecasts a change from the last observed value.

The stock TFT emits the target level in normalized space. On near-unit-root
prices that forces the network to rebuild today's price from a normalizer
fitted once on the training years, and small errors in that reconstruction
cost it the short horizons. Here the network output is added to the
normalized last encoder value before de-normalization, so an all-zero output
reproduces the persistence forecast exactly and the network only learns
departures from it.
"""
from pytorch_forecasting import TemporalFusionTransformer


class AnchoredTFT(TemporalFusionTransformer):

    def forward(self, x):
        # raw-scale last observed value; encoder lengths are fixed, so the
        # final encoder position is the forecast origin for every sample
        self._anchor = x["encoder_target"][:, -1]
        try:
            return super().forward(x)
        finally:
            self._anchor = None

    def transform_output(self, prediction, target_scale, loss=None):
        anchor = getattr(self, "_anchor", None)
        if anchor is not None:
            # same transformation, center and scale the normalizer applies to the target
            center, scale = target_scale[..., 0], target_scale[..., 1]
            anchor_norm = (self.output_transformer.preprocess(anchor) - center) / scale
            prediction = prediction + anchor_norm.view(-1, *([1] * (prediction.ndim - 1)))
        return super().transform_output(prediction, target_scale, loss)
