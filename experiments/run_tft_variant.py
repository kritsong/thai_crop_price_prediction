"""Train one TFT variant under the published protocol and save its forecasts.

Two changes can be switched on independently of the loss decay exponent.

--scale-weighted  weights every window by median_price / product_price, both
                  2018-2022 means, so the quantile loss (which pytorch-forecasting
                  evaluates in price units) is no longer dominated by the most
                  expensive products.
--anchored        uses AnchoredTFT, whose output is a change from the last
                  observed value, so a zero output is the persistence forecast.

Everything else (data, splits, architecture, optimizer, budget) matches
src/models/train_tft.py. Outputs go to EXPERIMENTS_DIR/variants and never
touch the published artifacts.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse, time, warnings
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer

from paths import EXPERIMENTS_DIR
from src.data.tft_loader import build_dataset
from src.models.loss import HorizonWeightedQuantileLoss
from src.models.anchored_tft import AnchoredTFT
from src.models.train_tft import collect_forecasts, time_idx_for_date, mae_np, HORIZONS

warnings.filterwarnings("ignore")


class EpochTimer(pl.Callback):
    """Log each epoch and keep an in-memory copy of the lowest-validation-loss weights."""

    def __init__(self):
        self.t0 = time.time()
        self.best = (float("inf"), None, None)

    def on_validation_epoch_end(self, trainer, module):
        if trainer.sanity_checking:
            return
        loss = float(trainer.callback_metrics["val_loss"])
        epoch = trainer.current_epoch + 1
        if loss < self.best[0]:
            state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
            self.best = (loss, epoch, state)
        print(f"[epoch {epoch}] val_loss={loss:.6g}  elapsed {(time.time() - self.t0) / 60:.1f} min", flush=True)


def check_anchor_identity(model, loader):
    """With the output layer zeroed, every quantile at every step must equal the last observed value."""
    saved = {k: v.clone() for k, v in model.output_layer.state_dict().items()}
    for p in model.output_layer.parameters():
        torch.nn.init.zeros_(p)
    model.eval()
    x, _ = next(iter(loader))
    x = {k: (v.to(model.device) if isinstance(v, torch.Tensor) else v) for k, v in x.items()}
    with torch.no_grad():
        pred = model(x).prediction
    last = x["encoder_target"][:, -1].view(-1, 1, 1).expand_as(pred)
    err = (pred - last).abs().max().item()
    model.output_layer.load_state_dict(saved)
    model.train()
    print(f"anchor identity check: max |zero-output forecast - last value| = {err:.3g}")
    # float32 round trip through softplus and the normalizer, relative to the largest price
    assert err < 1e-4 * max(1.0, last.abs().max().item()), "anchored output does not reproduce persistence"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--anchored", action="store_true")
    ap.add_argument("--scale-weighted", action="store_true")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--restore-best", action="store_true",
                    help="score the epoch with the lowest validation loss instead of the last one")
    ap.add_argument("--smoke", action="store_true", help="tiny run to exercise the code path")
    args = ap.parse_args()

    tag = (f"g{str(args.gamma).replace('.', '_')}_s{args.seed}_e{args.epochs}"
           f"{'_anch' if args.anchored else ''}{'_sw' if args.scale_weighted else ''}"
           f"{'' if args.lr == 0.01 else '_lr' + format(args.lr, 'g')}{'_best' if args.restore_best else ''}"
           f"{'_smoke' if args.smoke else ''}")
    out_dir = EXPERIMENTS_DIR / "variants"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"variant {tag}", flush=True)

    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")

    full_df, _ = build_dataset()
    max_prediction_length, max_encoder_length = 250, 30
    train_cutoff_idx = time_idx_for_date(full_df, "2023-01-02") - 1
    val_start_idx = time_idx_for_date(full_df, "2023-01-02")
    test_start_idx = time_idx_for_date(full_df, "2024-01-01")

    weight_col = None
    if args.scale_weighted:
        level = full_df[full_df.time_idx <= train_cutoff_idx].groupby("product_id", observed=True)["price_min"].mean()
        w = level.median() / level
        full_df["w"] = full_df["product_id"].map(w).astype(float)
        weight_col = "w"
        print(f"scale weights: min {w.min():.3g}, median {w.median():.3g}, max {w.max():.3g}")

    train_frame = full_df[full_df["time_idx"] <= train_cutoff_idx]
    val_frame = full_df[full_df["time_idx"] < test_start_idx]

    training = TimeSeriesDataSet(
        train_frame, time_idx="time_idx", target="price_min", group_ids=["product_id"], weight=weight_col,
        min_encoder_length=max_encoder_length, max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length, max_prediction_length=max_prediction_length,
        static_categoricals=["product_id", "category_name", "group_name"],
        time_varying_known_reals=["month", "day_of_week", "day_of_year", "time_idx"],
        time_varying_unknown_reals=["price_min"],
        target_normalizer=GroupNormalizer(groups=["product_id"], transformation="softplus"),
        add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
    )
    validation = TimeSeriesDataSet.from_dataset(training, val_frame, predict=False,
                                                stop_randomization=True, min_prediction_idx=val_start_idx)
    valfit = TimeSeriesDataSet.from_dataset(
        training, val_frame, predict=False, stop_randomization=True,
        min_prediction_idx=max(val_start_idx - (max_prediction_length - 1), max_encoder_length))
    test = TimeSeriesDataSet.from_dataset(training, full_df, predict=False,
                                          stop_randomization=True, min_prediction_idx=test_start_idx)

    bs = 64
    kw = dict(num_workers=args.workers, persistent_workers=args.workers > 0)
    train_dl = training.to_dataloader(train=True, batch_size=bs, **kw)
    val_dl = validation.to_dataloader(train=False, batch_size=bs * 2, **kw)
    valfit_dl = valfit.to_dataloader(train=False, batch_size=bs * 2, **kw)
    test_dl = test.to_dataloader(train=False, batch_size=bs * 2, **kw)

    cls = AnchoredTFT if args.anchored else TemporalFusionTransformer
    tft = cls.from_dataset(
        training, learning_rate=args.lr, hidden_size=64, attention_head_size=4, dropout=0.1,
        hidden_continuous_size=8, loss=HorizonWeightedQuantileLoss(gamma=args.gamma),
        log_interval=10, reduce_on_plateau_patience=4,
    )
    if args.anchored:
        check_anchor_identity(tft, val_dl)

    timer = EpochTimer()
    trainer = pl.Trainer(
        max_epochs=args.epochs, accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1,
        gradient_clip_val=0.1,
        callbacks=[EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, mode="min"), timer],
        enable_checkpointing=False, logger=False, enable_progress_bar=False,
        limit_train_batches=20 if args.smoke else 1.0, limit_val_batches=5 if args.smoke else 1.0,
    )
    trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)
    if args.restore_best:
        best_loss, best_epoch, best_state = timer.best
        tft.load_state_dict(best_state)
        print(f"restored epoch {best_epoch} (val_loss {best_loss:.6g})", flush=True)
    tft.eval()

    stores = {"val": collect_forecasts(tft, valfit_dl), "test": collect_forecasts(tft, test_dl)}
    arrays = {f"{split}_h{h}_{k}": v for split, st in stores.items() for h in HORIZONS for k, v in st[h].items()}
    np.savez_compressed(out_dir / f"{tag}.npz", **arrays)
    torch.save(tft.state_dict(), out_dir / f"{tag}.pt")

    for h in [20, 60, 120, 250]:
        s = stores["test"][h]
        print(f"test h={h:3d}: MAE {mae_np(s['actual'], s['median']):8.2f}  persistence {mae_np(s['actual'], s['persistence']):8.2f}")
    print("saved", out_dir / f"{tag}.npz", flush=True)


if __name__ == "__main__":
    main()
