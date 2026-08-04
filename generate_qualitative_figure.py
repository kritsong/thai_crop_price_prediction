"""Qualitative forecast figure, four products, from the gamma=4.5 checkpoint.

Loads the trained network and runs a forward pass from the earliest 2024
origin, then plots the quantile forecast against what actually happened.

Two of the four products are carried over from earlier drafts. The other two
were picked on purpose to keep this honest: one where we win big at long
horizon, and one where we lose even at a year out.
"""
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from src.models.loss import HorizonWeightedQuantileLoss
from src.data.tft_loader import build_dataset

from paths import CHAMPION_CKPT, IMAGE_DIR

CKPT = str(CHAMPION_CKPT)
OUT = str(IMAGE_DIR / "qualitative_predictions.png")

PRODUCTS = [
    ("R11031", "Jasmine rice 100% grade 2, wholesale", "Wholesale Rice & Bags"),
    ("W18098", "Soybean oil, wholesale", "Oil Crops & Oils"),
    ("R11033", "Broken jasmine rice, wholesale", "Wholesale Rice & Bags"),
    ("W11049", "Fresh chicken (legs, feet), wholesale", "Meat & Poultry"),
]


def time_idx_for_date(full_df, date):
    mask = pd.to_datetime(full_df["date"]) >= pd.to_datetime(date)
    return int(full_df.loc[mask, "time_idx"].min())


def main():
    torch.set_float32_matmul_precision("medium")
    full_df, _ = build_dataset()

    max_prediction_length = 250
    max_encoder_length = 30
    train_cutoff_idx = time_idx_for_date(full_df, "2023-01-02") - 1
    test_start_idx = time_idx_for_date(full_df, "2024-01-01")
    train_frame = full_df[full_df["time_idx"] <= train_cutoff_idx]

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
        target_normalizer=GroupNormalizer(groups=["product_id"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.01,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=HorizonWeightedQuantileLoss(gamma=4.5),
    )
    state = torch.load(CKPT, map_location=device)
    tft.load_state_dict(state)
    tft.to(device)
    tft.eval()
    print("Loaded champion checkpoint:", CKPT)

    pids = [p[0] for p in PRODUCTS]
    sub_df = full_df[full_df["product_id"].astype(str).isin(pids)].copy()

    # stop_randomization=True with only a floor on min_prediction_idx yields EVERY
    # valid origin at/after that floor, not one window per group; scan the whole
    # dataloader and keep, per product, only the window with the earliest origin
    # (closest to 2024-01-02), analogous to predict=True but pinned to that date.
    infer_ds = TimeSeriesDataSet.from_dataset(
        training, sub_df, predict=False, stop_randomization=True,
        min_prediction_idx=test_start_idx,
    )
    dl = infer_ds.to_dataloader(train=False, batch_size=64, shuffle=False, num_workers=0)

    date_map = full_df.drop_duplicates("time_idx").set_index("time_idx")["date"]
    group_encoder = training._categorical_encoders["__group_id__product_id"]

    best_origin = {}
    results = {}
    with torch.no_grad():
        for batch in dl:
            x, y = batch
            for k, v in x.items():
                if isinstance(v, torch.Tensor):
                    x[k] = v.to(device)
            out = tft(x)
            pred = out.prediction.cpu().numpy()          # (batch, 250, 3)
            enc_target = x["encoder_target"].cpu().numpy()
            dec_time_idx = x["decoder_time_idx"].cpu().numpy()
            enc_len = enc_target.shape[1]
            enc_time_idx = dec_time_idx[:, :1] - enc_len + np.arange(enc_len)[None, :]
            actual = y[0].cpu().numpy()
            groups = x["groups"][:, 0].cpu().numpy()
            pid_batch = group_encoder.inverse_transform(groups)

            for i, pid in enumerate(pid_batch):
                if pid not in pids:
                    continue
                origin = int(dec_time_idx[i, 0])
                if pid in best_origin and origin >= best_origin[pid]:
                    continue
                best_origin[pid] = origin
                results[pid] = {
                    "lookback_dates": date_map.loc[enc_time_idx[i]].to_numpy(),
                    "lookback_actual": enc_target[i],
                    "future_dates": date_map.loc[dec_time_idx[i]].to_numpy(),
                    "future_actual": actual[i],
                    "lower": pred[i, :, 0],
                    "median": pred[i, :, 1],
                    "upper": pred[i, :, 2],
                }

    missing = [p for p in pids if p not in results]
    if missing:
        raise RuntimeError(f"Missing inference results for {missing}")
    print("Selected origins (time_idx):", best_origin)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
    for ax, (pid, label, group) in zip(axes.ravel(), PRODUCTS):
        r = results[pid]
        ax.plot(pd.to_datetime(r["lookback_dates"]), r["lookback_actual"],
                 color="#222222", linewidth=1.6, label="Lookback (actual)")
        ax.plot(pd.to_datetime(r["future_dates"]), r["future_actual"],
                 color="#2166ac", linewidth=1.6, label="Held-out (actual)")
        ax.plot(pd.to_datetime(r["future_dates"]), r["median"],
                 color="#b2182b", linewidth=1.4, linestyle="--", label="Median forecast")
        ax.fill_between(pd.to_datetime(r["future_dates"]), r["lower"], r["upper"],
                         color="#b2182b", alpha=0.15, label="80% interval")
        ax.axvline(pd.to_datetime(r["future_dates"][0]), color="#555555", linewidth=0.8, linestyle=":")
        ax.set_title(f"{pid}: {label}\n({group})", loc="left", fontsize=9.5, fontweight="bold")
        ax.set_ylabel("Price (THB)")
        ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.8)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", labelrotation=30)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=4, frameon=False)
    fig.suptitle("Selected model forecasts on representative held-out trajectories (real inference)",
                 y=1.08, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print("Saved", OUT)


if __name__ == "__main__":
    main()
