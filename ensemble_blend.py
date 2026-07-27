"""Post-hoc gamma-ensemble + persistence blend evaluation.

Loads the per-sample predictions saved by each gamma run of
src/models/train_tft.py, averages the median forecasts across gammas,
re-fits the per-horizon persistence-blend weight and the conformal PI scale
on the 2023 validation labels, and evaluates on the 2024-2025 test set.

Writes results/ensemble_blend_metrics.csv. No retraining required.
"""
import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent
from paths import EXPERIMENTS_DIR

RESULTS_DIR = EXPERIMENTS_DIR
OUT_PATH = PACKAGE_ROOT / "results" / "ensemble_blend_metrics.csv"

HORIZONS = [1, 20, 60, 120, 250]
# Mon-Fri skeleton: business days in the 2023 validation year
VAL_YEAR_BDAYS = int(np.busday_count("2023-01-01", "2024-01-01"))


def smape_np(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred)
    return float(np.mean(np.where(denom == 0, 0.0, 200.0 * np.abs(y_pred - y_true) / (denom + 1e-8))))


def mae_np(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse_np(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def gamma_of(path):
    # matches only seed-42 archives (e.g. gamma_2_0_predictions.npz);
    # seed-tagged replicates (gamma_3_5_s43_predictions.npz) return None
    m = re.search(r"gamma_(\d+)_(\d+)_predictions", path)
    return float(f"{m.group(1)}.{m.group(2)}") if m else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-gamma", type=float, default=3.0,
        help="ensemble membership cutoff; composition is selected on validation "
             "(moderate exponents gamma<=3.0 win; steeper members degrade interval quality)",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(str(RESULTS_DIR / "tft_hw_quantile_gamma_*_predictions.npz")))
    files = [f for f in files if gamma_of(f) is not None and gamma_of(f) <= args.max_gamma]
    if not files:
        raise SystemExit("no prediction archives found - run the gamma sweep first")
    print(f"Loading {len(files)} prediction archives (gamma <= {args.max_gamma}):")
    for f in files:
        print(f"  {f}")
    runs = [np.load(f) for f in files]

    # sample alignment across runs (dataloaders are deterministic for train=False)
    ref = runs[0]
    for r in runs[1:]:
        for h in HORIZONS:
            for split in ("val", "test"):
                assert np.array_equal(ref[f"{split}_h{h}_label_idx"], r[f"{split}_h{h}_label_idx"]), "sample misalignment"
                assert np.array_equal(ref[f"{split}_h{h}_group"], r[f"{split}_h{h}_group"]), "sample misalignment"

    test_start_idx = int(ref["test_h1_label_idx"].min())
    val_start_idx = test_start_idx - VAL_YEAR_BDAYS
    print(f"test_start_idx={test_start_idx}, val_start_idx={val_start_idx}")

    rows = []
    for h in HORIZONS:
        ens = {}
        for split in ("val", "test"):
            ens[split] = {
                "actual": ref[f"{split}_h{h}_actual"],
                "persistence": ref[f"{split}_h{h}_persistence"],
                "label_idx": ref[f"{split}_h{h}_label_idx"],
                "median": np.mean([r[f"{split}_h{h}_median"] for r in runs], axis=0),
                "lower": np.mean([r[f"{split}_h{h}_lower"] for r in runs], axis=0),
                "upper": np.mean([r[f"{split}_h{h}_upper"] for r in runs], axis=0),
            }

        v = ens["val"]
        mask = v["label_idx"] >= val_start_idx
        actual, median, persistence = v["actual"][mask], v["median"][mask], v["persistence"][mask]

        grid = np.linspace(0.0, 1.0, 101)
        maes = [mae_np(actual, a * median + (1 - a) * persistence) for a in grid]
        alpha = float(grid[int(np.argmin(maes))])

        eps = 1e-8
        lo_dist = np.maximum(median - v["lower"][mask], eps)
        hi_dist = np.maximum(v["upper"][mask] - median, eps)
        r_scores = np.maximum((median - actual) / lo_dist, (actual - median) / hi_dist)
        scale = float(np.quantile(r_scores, 0.8))

        t = ens["test"]
        blend = alpha * t["median"] + (1 - alpha) * t["persistence"]
        cal_lower = t["median"] - scale * (t["median"] - t["lower"])
        cal_upper = t["median"] + scale * (t["upper"] - t["median"])
        cal_covered = (t["actual"] >= cal_lower) & (t["actual"] <= cal_upper)

        row = {
            "Horizon": h,
            "Ensemble_MAE": mae_np(t["actual"], t["median"]),
            "Ensemble_SMAPE": smape_np(t["actual"], t["median"]),
            "Ensemble_RMSE": rmse_np(t["actual"], t["median"]),
            "Blend_Alpha": alpha,
            "Blend_MAE": mae_np(t["actual"], blend),
            "Blend_SMAPE": smape_np(t["actual"], blend),
            "Blend_RMSE": rmse_np(t["actual"], blend),
            "Baseline_MAE": mae_np(t["actual"], t["persistence"]),
            "Baseline_SMAPE": smape_np(t["actual"], t["persistence"]),
            "Baseline_RMSE": rmse_np(t["actual"], t["persistence"]),
            "CalPIScale": scale,
            "CalPIWidth": float(np.mean(cal_upper - cal_lower)),
            "CalPICoverage": float(np.mean(cal_covered)),
            "ValFitSamples": int(mask.sum()),
        }
        rows.append(row)
        beats = "YES" if row["Blend_MAE"] < row["Baseline_MAE"] else "no"
        print(
            f"H{h}: baseline MAE={row['Baseline_MAE']:.4f} | ensemble MAE={row['Ensemble_MAE']:.4f} | "
            f"blend(a={alpha:.2f}) MAE={row['Blend_MAE']:.4f} | beats baseline: {beats} | "
            f"cal 80% PI cov={row['CalPICoverage']:.3f} width={row['CalPIWidth']:.2f}"
        )

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    df.to_csv(RESULTS_DIR / "ensemble_blend_metrics.csv", index=False)
    main_h = df[df.Horizon.isin([20, 60, 120, 250])]
    print(f"\nOverall (H20/60/120/250): baseline MAE={main_h.Baseline_MAE.mean():.4f} | "
          f"ensemble MAE={main_h.Ensemble_MAE.mean():.4f} | blend MAE={main_h.Blend_MAE.mean():.4f}")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
