"""Regenerate Figures 8-11 (attention, static/encoder/decoder variable selection)
from the actual champion checkpoint (gamma=4.5, seed 42), rather than the old
June training run these figures previously came from.
"""
import sys
import warnings
warnings.filterwarnings("ignore")

import torch
import matplotlib
matplotlib.use("Agg")

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
# make the repository root importable regardless of how this is invoked
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from src.models.loss import HorizonWeightedQuantileLoss
from src.data.tft_loader import build_dataset

import pandas as pd

from paths import CHAMPION_CKPT, IMAGE_DIR as _IMAGE_DIR

CKPT = str(CHAMPION_CKPT)
IMAGE_DIR = str(_IMAGE_DIR)


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
        train_frame, time_idx="time_idx", target="price_min", group_ids=["product_id"],
        min_encoder_length=max_encoder_length, max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length, max_prediction_length=max_prediction_length,
        static_categoricals=["product_id", "category_name", "group_name"],
        time_varying_known_reals=["month", "day_of_week", "day_of_year", "time_idx"],
        time_varying_unknown_reals=["price_min"],
        target_normalizer=GroupNormalizer(groups=["product_id"], transformation="softplus"),
        add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
    )
    test = TimeSeriesDataSet.from_dataset(
        training, full_df, predict=False, stop_randomization=True, min_prediction_idx=test_start_idx,
    )
    test_dataloader = test.to_dataloader(train=False, batch_size=128, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tft = TemporalFusionTransformer.from_dataset(
        training, learning_rate=0.01, hidden_size=64, attention_head_size=4,
        dropout=0.1, hidden_continuous_size=8, loss=HorizonWeightedQuantileLoss(gamma=4.5),
    )
    tft.load_state_dict(torch.load(CKPT, map_location=device))
    tft.to(device)
    tft.eval()
    print("Loaded champion checkpoint:", CKPT)

    x, y = next(iter(test_dataloader))
    for k, v in x.items():
        if isinstance(v, torch.Tensor):
            x[k] = v.to(device)

    with torch.no_grad():
        out = tft(x)
        interpretation = tft.interpret_output(out, reduction="sum")

    figs = tft.plot_interpretation(interpretation)
    figs["attention"].savefig(f"{IMAGE_DIR}/tft_attention_ablation.png", bbox_inches="tight")
    figs["static_variables"].savefig(f"{IMAGE_DIR}/tft_static_vars_ablation.png", bbox_inches="tight")
    figs["encoder_variables"].savefig(f"{IMAGE_DIR}/tft_encoder_vars_ablation.png", bbox_inches="tight")
    figs["decoder_variables"].savefig(f"{IMAGE_DIR}/tft_decoder_vars_ablation.png", bbox_inches="tight")
    print("Saved interpretability figures from the champion (gamma=4.5) checkpoint.")

    def show_pct(label, names, raw):
        raw = raw.detach().cpu().numpy()
        pct = 100 * raw / raw.sum()
        print(f"\n=== {label} importance (%) ===")
        for i in pct.argsort()[::-1]:
            print(f"  {names[i]}: {pct[i]:.2f}%")

    show_pct("Static variable", tft.static_variables, interpretation["static_variables"])
    show_pct("Encoder variable", tft.encoder_variables, interpretation["encoder_variables"])
    show_pct("Decoder variable", tft.decoder_variables, interpretation["decoder_variables"])

    print("\n=== Attention: peak location and shape ===")
    att = interpretation["attention"].detach().cpu().numpy()
    print("  attention array length:", len(att))
    print("  argmax index (0=oldest of 30-day encoder):", int(att.argmax()), "value:", float(att.max()))
    print("  value at index 0 (oldest):", float(att[0]))
    print("  value at last index (most recent):", float(att[-1]))
    print("  mean of remaining (excluding argmax):", float((att.sum() - att.max()) / (len(att) - 1)))


if __name__ == "__main__":
    main()
