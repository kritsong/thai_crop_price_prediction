import warnings
import os
import matplotlib; matplotlib.use('Agg')
import pandas as pd
import numpy as np
import optuna
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
import torch

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from pytorch_forecasting.data import GroupNormalizer
from optuna.integration import PyTorchLightningPruningCallback

from src.data.tft_loader import build_dataset
from paths import RESULTS_DIR

warnings.filterwarnings("ignore")

def time_idx_for_date(full_df, date):
    mask = pd.to_datetime(full_df["date"]) >= pd.to_datetime(date)
    if not mask.any():
        raise ValueError(f"No time_idx found on or after {date}")
    return int(full_df.loc[mask, "time_idx"].min())

def objective(trial):
    pl.seed_everything(42)
    
    full_df, _ = build_dataset()
    
    max_prediction_length = 250
    max_encoder_length = int(os.environ.get("TFT_ENCODER_LENGTH", "30"))
    
    train_cutoff_idx = time_idx_for_date(full_df, "2023-01-02") - 1
    val_start_idx = time_idx_for_date(full_df, "2023-01-02")
    test_start_idx = time_idx_for_date(full_df, "2024-01-01")
    train_frame = full_df[full_df["time_idx"] <= train_cutoff_idx]
    val_frame = full_df[full_df["time_idx"] < test_start_idx]

    # Create dataset and dataloaders
    training_ds = TimeSeriesDataSet(
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

    validation_ds = TimeSeriesDataSet.from_dataset(
        training_ds,
        val_frame,
        predict=False,
        stop_randomization=True,
        min_prediction_idx=val_start_idx,
    )

    batch_size = 64
    train_dataloader = training_ds.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation_ds.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=0)

    # Optuna sampling
    hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64])
    attention_head_size = trial.suggest_categorical("attention_head_size", [1, 2, 4])
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    hidden_continuous_size = trial.suggest_categorical("hidden_continuous_size", [8, 16, 32])
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.1, log=True)

    # Model
    tft = TemporalFusionTransformer.from_dataset(
        training_ds,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    # Trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=3, verbose=False, mode="min")
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")

    trainer = pl.Trainer(
        max_epochs=3, # Keep tuning short to fit budget
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback, pruning_callback],
        enable_progress_bar=False,
        logger=False
    )

    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    return trainer.callback_metrics["val_loss"].item()

if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="minimize", pruner=pruner)
    print("Starting Optuna TFT Hyperparameter Tuning (10 Trials)...")
    study.optimize(objective, n_trials=10)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(
        RESULTS_DIR / "tft_architecture_search.csv", index=False
    )

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    print("Saved trial ledger to", RESULTS_DIR / "tft_architecture_search.csv")
