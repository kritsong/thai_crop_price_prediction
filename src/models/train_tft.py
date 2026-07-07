import matplotlib
matplotlib.use('Agg')
import os
import glob
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss, MAE, MultiHorizonMetric
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from sklearn.metrics import mean_absolute_error
import torch.nn.functional as F
import math
import argparse

from src.models.loss import HorizonWeightedQuantileLoss





from src.data.tft_loader import build_dataset

warnings.filterwarnings("ignore")
HORIZONS = [1, 20, 60, 120, 250]

RESULTS_DIR = Path("d:/new_crop_data/experiments_results")
IMAGE_DIR = Path("d:/new_crop_data/paper/images")

def smape_np(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred)
    return np.mean(np.where(denom == 0, 0.0, 200.0 * np.abs(y_pred - y_true) / (denom + 1e-8)))

def time_idx_for_date(full_df, date):
    mask = pd.to_datetime(full_df["date"]) >= pd.to_datetime(date)
    if not mask.any():
        raise ValueError(f"No time_idx found on or after {date}")
    return int(full_df.loc[mask, "time_idx"].min())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=2.0, help="Gamma parameter for HorizonWeightedQuantileLoss")
    args = parser.parse_args()

    pl.seed_everything(42)
    torch.set_float32_matmul_precision("medium")
    
    full_df, _ = build_dataset()
    
    max_prediction_length = 250
    max_encoder_length = int(os.environ.get("TFT_ENCODER_LENGTH", "30"))
    
    train_cutoff_idx = time_idx_for_date(full_df, "2023-01-02") - 1
    val_start_idx = time_idx_for_date(full_df, "2023-01-02")
    test_start_idx = time_idx_for_date(full_df, "2024-01-01")
    
    train_frame = full_df[full_df["time_idx"] <= train_cutoff_idx]
    val_frame = full_df[full_df["time_idx"] < test_start_idx]

    print(f"TFT encoder length: {max_encoder_length}")
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
    test_dataloader = test.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=0)
    
    print("Initializing TFT Model...")
    # Create the model using the optimal Optuna parameters
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

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[lr_logger, early_stop_callback],
        enable_checkpointing=False,
        logger=False,
    )

    print("Training TFT...")
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )
    
    print("Evaluating TFT on Test Set Horizons (Streaming)...")
    tft.eval()
    abs_errors = {h: [] for h in HORIZONS}
    sq_errors = {h: [] for h in HORIZONS}
    smape_vals = {h: [] for h in HORIZONS}
    indices = [h - 1 for h in HORIZONS]
    
    with torch.no_grad():
        for batch in test_dataloader:
            x, y = batch
            # Move x to device
            for k, v in x.items():
                if isinstance(v, torch.Tensor):
                    x[k] = v.to(tft.device)
            
            # Forward pass
            out = tft(x)
            
            # HorizonWeightedQuantileLoss outputs shape (batch, max_prediction_length, 3)
            # quantiles: [0.1, 0.5, 0.9]
            # y_pred_median is index 1
            y_pred_median = out.prediction[..., 1].cpu().numpy()
            y_pred_lower = out.prediction[..., 0].cpu().numpy()
            y_pred_upper = out.prediction[..., 2].cpu().numpy()
            y_true = y[0].cpu().numpy()
            
            for h, idx in zip(HORIZONS, indices):
                err = y_true[:, idx] - y_pred_median[:, idx]
                abs_errors[h].extend(np.abs(err))
                sq_errors[h].extend(err ** 2)
                
                # 80% PI Coverage
                covered = (y_true[:, idx] >= y_pred_lower[:, idx]) & (y_true[:, idx] <= y_pred_upper[:, idx])
                if "pi_coverage" not in locals():
                    pi_coverage = {h_k: [] for h_k in HORIZONS}
                pi_coverage[h].extend(covered)
                
                smape_vals[h].extend(
                    np.where(
                        np.abs(y_true[:, idx]) + np.abs(y_pred_median[:, idx]) == 0,
                        0.0,
                        200.0 * np.abs(err) / (np.abs(y_true[:, idx]) + np.abs(y_pred_median[:, idx]) + 1e-8)
                    )
                )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    for h in HORIZONS:
        mae_h = float(np.mean(abs_errors[h]))
        rmse_h = float(np.sqrt(np.mean(sq_errors[h])))
        smape_h = float(np.mean(smape_vals[h]))
        cov_h = float(np.mean(pi_coverage[h])) if "pi_coverage" in locals() else 0.0
        metric_rows.append({"Horizon": h, "SMAPE": smape_h, "MAE": mae_h, "RMSE": rmse_h, "PICoverage": cov_h})
        print(f"TFT H{h}: SMAPE={smape_h:.4f}, MAE={mae_h:.4f}, RMSE={rmse_h:.4f}, 80% PI Cov={cov_h:.4f}")
        
    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = RESULTS_DIR / "tft_strict_pure_price_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved strict pure-price TFT metrics to {metrics_path}")
    print(f"TFT Overall Mean MAE: {metrics_df['MAE'].mean():.4f}")

    # Interpretability Extraction (on a single batch to avoid OOM)
    print("Extracting TFT Interpretability Metrics...")
    x, y = next(iter(test_dataloader))
    for k, v in x.items():
        if isinstance(v, torch.Tensor):
            x[k] = v.to(tft.device)
    
    out = tft(x)
    interpretation = tft.interpret_output(out, reduction="sum")
    
    import matplotlib.pyplot as plt
    figs = tft.plot_interpretation(interpretation)
    
    # Save figures to the artifact directory
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    figs['attention'].savefig(IMAGE_DIR / "tft_attention_ablation.png", bbox_inches="tight")
    figs['static_variables'].savefig(IMAGE_DIR / "tft_static_vars_ablation.png", bbox_inches="tight")
    figs['encoder_variables'].savefig(IMAGE_DIR / "tft_encoder_vars_ablation.png", bbox_inches="tight")
    figs['decoder_variables'].savefig(IMAGE_DIR / "tft_decoder_vars_ablation.png", bbox_inches="tight")
    print("Saved interpretability artifacts.")

if __name__ == "__main__":
    main()
