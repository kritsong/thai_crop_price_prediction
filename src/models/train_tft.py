import matplotlib
matplotlib.use('Agg')
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from lightning.pytorch.callbacks import EarlyStopping
import argparse

from src.models.loss import HorizonWeightedQuantileLoss
from src.data.tft_loader import build_dataset

warnings.filterwarnings("ignore")
HORIZONS = [1, 20, 60, 120, 250]

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
from paths import EXPERIMENTS_DIR, IMAGE_DIR

RESULTS_DIR = EXPERIMENTS_DIR


def smape_np(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred)
    return float(np.mean(np.where(denom == 0, 0.0, 200.0 * np.abs(y_pred - y_true) / (denom + 1e-8))))


def mae_np(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse_np(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def time_idx_for_date(full_df, date):
    mask = pd.to_datetime(full_df["date"]) >= pd.to_datetime(date)
    if not mask.any():
        raise ValueError(f"No time_idx found on or after {date}")
    return int(full_df.loc[mask, "time_idx"].min())


def collect_forecasts(tft, dataloader):
    """Run inference and collect, per horizon: actuals, quantile forecasts, the
    Lag-1 persistence forecast (last observed encoder value), the label time_idx
    and the product group id on the exact same samples so model-vs-baseline
    comparisons are apples-to-apples."""
    keys = ["actual", "median", "lower", "upper", "persistence", "label_idx", "group"]
    store = {h: {k: [] for k in keys} for h in HORIZONS}
    indices = [h - 1 for h in HORIZONS]

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            for k, v in x.items():
                if isinstance(v, torch.Tensor):
                    x[k] = v.to(tft.device)

            out = tft(x)

            # quantiles [0.1, 0.5, 0.9] -> (batch, max_prediction_length, 3)
            y_pred_lower = out.prediction[..., 0].cpu().numpy()
            y_pred_median = out.prediction[..., 1].cpu().numpy()
            y_pred_upper = out.prediction[..., 2].cpu().numpy()
            y_true = y[0].cpu().numpy()
            # Lag-1 persistence forecast for every decoder step of this window
            y_last = x["encoder_target"][:, -1].cpu().numpy()
            label_idx = x["decoder_time_idx"].cpu().numpy()
            group = x["groups"][:, 0].cpu().numpy()

            for h, idx in zip(HORIZONS, indices):
                store[h]["actual"].extend(y_true[:, idx])
                store[h]["median"].extend(y_pred_median[:, idx])
                store[h]["lower"].extend(y_pred_lower[:, idx])
                store[h]["upper"].extend(y_pred_upper[:, idx])
                store[h]["persistence"].extend(y_last)
                store[h]["label_idx"].extend(label_idx[:, idx])
                store[h]["group"].extend(group)

    for h in HORIZONS:
        for k in store[h]:
            store[h][k] = np.asarray(store[h][k], dtype=np.float64)
    return store


def fit_blend_alphas(val_store, val_start_idx):
    """Fit, per horizon, the convex blend weight alpha in
    y_blend = alpha * y_tft_median + (1 - alpha) * y_persistence
    minimizing MAE over all windows whose label at that horizon falls in the
    validation year. Restricting on the label (not the origin) keeps far
    horizons from being starved down to a handful of January origins."""
    alphas = {}
    grid = np.linspace(0.0, 1.0, 101)
    for h in HORIZONS:
        s = val_store[h]
        mask = s["label_idx"] >= val_start_idx
        actual, median, persistence = s["actual"][mask], s["median"][mask], s["persistence"][mask]
        print(f"alpha fit H{h}: {mask.sum()} validation samples")
        maes = [mae_np(actual, a * median + (1 - a) * persistence) for a in grid]
        alphas[h] = float(grid[int(np.argmin(maes))])
    return alphas


def fit_pi_scales(val_store, val_start_idx, target_coverage=0.8):
    """Fit, per horizon, a conformal-style scale factor s for the quantile
    interval so that [median - s*(median-lower), median + s*(upper-median)]
    achieves the target coverage on the validation year."""
    scales = {}
    eps = 1e-8
    for h in HORIZONS:
        s = val_store[h]
        mask = s["label_idx"] >= val_start_idx
        actual, median = s["actual"][mask], s["median"][mask]
        lo_dist = np.maximum(median - s["lower"][mask], eps)
        hi_dist = np.maximum(s["upper"][mask] - median, eps)
        # per-sample scale needed to cover the actual
        r = np.maximum((median - actual) / lo_dist, (actual - median) / hi_dist)
        scales[h] = float(np.quantile(r, target_coverage))
    return scales


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=2.0, help="Gamma for HorizonWeightedQuantileLoss w(h)=1/h^gamma")
    parser.add_argument("--seed", type=int, default=42, help="Random seed; non-default seeds get seed-tagged artifact filenames")
    parser.add_argument("--save-interpretation", action="store_true", help="Save interpretability PNGs (overwrites paper images)")
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")

    full_df, _ = build_dataset()

    max_prediction_length = 250
    max_encoder_length = int(os.environ.get("TFT_ENCODER_LENGTH", "30"))
    max_epochs = int(os.environ.get("TFT_MAX_EPOCHS", "5"))

    train_cutoff_idx = time_idx_for_date(full_df, "2023-01-02") - 1
    val_start_idx = time_idx_for_date(full_df, "2023-01-02")
    test_start_idx = time_idx_for_date(full_df, "2024-01-01")

    train_frame = full_df[full_df["time_idx"] <= train_cutoff_idx]
    val_frame = full_df[full_df["time_idx"] < test_start_idx]

    print(f"TFT encoder length: {max_encoder_length}, gamma: {args.gamma}, seed: {args.seed}, max epochs: {max_epochs}")
    print("Training labels end before 2023-01-02; validation starts 2023-01-02; test starts 2024-01-01.")

    training = TimeSeriesDataSet(
        train_frame,
        time_idx="time_idx",
        target="price_min",
        group_ids=["product_id"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["product_id", "category_name", "group_name"],
        time_varying_known_reals=["month", "day_of_week", "day_of_year", "time_idx"],
        time_varying_unknown_reals=["price_min"],
        target_normalizer=GroupNormalizer(
            groups=["product_id"], transformation="softplus"
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        val_frame,
        predict=False,
        stop_randomization=True,
        min_prediction_idx=val_start_idx,
    )
    # wider window for blend/calibration fitting: origins reach back into 2022 so
    # that every horizon has labels throughout the 2023 validation year; samples
    # are filtered per horizon on label_idx >= val_start_idx before fitting
    alphafit = TimeSeriesDataSet.from_dataset(
        training,
        val_frame,
        predict=False,
        stop_randomization=True,
        min_prediction_idx=max(val_start_idx - (max_prediction_length - 1), max_encoder_length),
    )
    test = TimeSeriesDataSet.from_dataset(
        training,
        full_df,
        predict=False,
        stop_randomization=True,
        min_prediction_idx=test_start_idx,
    )

    batch_size = 64
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=0)
    alphafit_dataloader = alphafit.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=0)
    test_dataloader = test.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=0)

    print("Initializing TFT Model...")
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.01,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=HorizonWeightedQuantileLoss(gamma=args.gamma),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    assert tft.hparams.output_size == 3, "expected a 3-quantile output head"

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback],
        enable_checkpointing=False,
        logger=False,
    )

    print("Training TFT...")
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

    tft.eval()

    print("Collecting blend-fit forecasts (labels in 2023)...")
    val_store = collect_forecasts(tft, alphafit_dataloader)
    blend_alphas = fit_blend_alphas(val_store, val_start_idx)
    pi_scales = fit_pi_scales(val_store, val_start_idx)
    print(f"Fitted blend alphas (validation MAE-optimal): {blend_alphas}")
    print(f"Fitted 80% PI conformal scales: { {h: round(s, 3) for h, s in pi_scales.items()} }")

    print("Evaluating on Test Set Horizons (2024-2025)...")
    test_store = collect_forecasts(tft, test_dataloader)

    metric_rows = []
    for h in HORIZONS:
        s = test_store[h]
        alpha = blend_alphas[h]
        blend = alpha * s["median"] + (1 - alpha) * s["persistence"]

        covered = (s["actual"] >= s["lower"]) & (s["actual"] <= s["upper"])
        scale = pi_scales[h]
        cal_lower = s["median"] - scale * (s["median"] - s["lower"])
        cal_upper = s["median"] + scale * (s["upper"] - s["median"])
        cal_covered = (s["actual"] >= cal_lower) & (s["actual"] <= cal_upper)
        row = {
            "Horizon": h,
            "Gamma": args.gamma,
            "Seed": args.seed,
            "SMAPE": smape_np(s["actual"], s["median"]),
            "MAE": mae_np(s["actual"], s["median"]),
            "RMSE": rmse_np(s["actual"], s["median"]),
            "PIWidth": float(np.mean(s["upper"] - s["lower"])),
            "PICoverage": float(np.mean(covered)),
            "CalPIScale": scale,
            "CalPIWidth": float(np.mean(cal_upper - cal_lower)),
            "CalPICoverage": float(np.mean(cal_covered)),
            "Baseline_SMAPE": smape_np(s["actual"], s["persistence"]),
            "Baseline_MAE": mae_np(s["actual"], s["persistence"]),
            "Baseline_RMSE": rmse_np(s["actual"], s["persistence"]),
            "Blend_Alpha": alpha,
            "Blend_SMAPE": smape_np(s["actual"], blend),
            "Blend_MAE": mae_np(s["actual"], blend),
            "Blend_RMSE": rmse_np(s["actual"], blend),
        }
        metric_rows.append(row)
        beats = "YES" if row["Blend_MAE"] < row["Baseline_MAE"] else "no"
        print(
            f"H{h}: baseline MAE={row['Baseline_MAE']:.4f} | TFT MAE={row['MAE']:.4f} | "
            f"blend(a={alpha:.2f}) MAE={row['Blend_MAE']:.4f} | beats baseline: {beats} | "
            f"80% PI cov={row['PICoverage']:.3f} -> calibrated {row['CalPICoverage']:.3f} (width {row['CalPIWidth']:.2f})"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(metric_rows)
    gamma_tag = str(args.gamma).replace(".", "_")
    if args.seed != 42:
        gamma_tag += f"_s{args.seed}"
    metrics_path = RESULTS_DIR / f"tft_hw_quantile_gamma_{gamma_tag}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to {metrics_path}")
    print(f"TFT Overall Mean MAE: {metrics_df['MAE'].mean():.4f} | Blend: {metrics_df['Blend_MAE'].mean():.4f} | Baseline: {metrics_df['Baseline_MAE'].mean():.4f}")

    # persist everything needed to refit blends/calibration without retraining
    torch.save(tft.state_dict(), RESULTS_DIR / f"tft_hw_quantile_gamma_{gamma_tag}.ckpt")
    pred_arrays = {}
    for name, store in [("val", val_store), ("test", test_store)]:
        for h in HORIZONS:
            for k, arr in store[h].items():
                pred_arrays[f"{name}_h{h}_{k}"] = arr
    np.savez_compressed(RESULTS_DIR / f"tft_hw_quantile_gamma_{gamma_tag}_predictions.npz", **pred_arrays)
    print("Saved checkpoint and per-sample predictions.")

    if args.save_interpretation:
        print("Extracting TFT Interpretability Metrics...")
        x, y = next(iter(test_dataloader))
        for k, v in x.items():
            if isinstance(v, torch.Tensor):
                x[k] = v.to(tft.device)

        out = tft(x)
        interpretation = tft.interpret_output(out, reduction="sum")
        figs = tft.plot_interpretation(interpretation)

        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        figs['attention'].savefig(IMAGE_DIR / "tft_attention_ablation.png", bbox_inches="tight")
        figs['static_variables'].savefig(IMAGE_DIR / "tft_static_vars_ablation.png", bbox_inches="tight")
        figs['encoder_variables'].savefig(IMAGE_DIR / "tft_encoder_vars_ablation.png", bbox_inches="tight")
        figs['decoder_variables'].savefig(IMAGE_DIR / "tft_decoder_vars_ablation.png", bbox_inches="tight")
        print("Saved interpretability artifacts.")


if __name__ == "__main__":
    main()
