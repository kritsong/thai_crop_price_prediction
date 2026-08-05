"""Rebuild the selected-model tables and model-selection audit trail.

The large per-window ``.npz`` files remain outside version control.  This
script reduces them to the version-controlled CSVs used by the paper and writes
cryptographic provenance for the champion artifact.  No model is retrained.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# make the repository root importable regardless of how this is invoked
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import CHAMPION_NPZ, EXPERIMENTS_DIR, RESULTS_DIR


HORIZONS = (20, 60, 120, 250)
EXPECTED_GAMMAS = tuple(i / 2 for i in range(17))
DATE_INDEX = pd.bdate_range("2018-01-03", "2025-12-30")
PRIMARY_NPZ = re.compile(
    r"^tft_hw_quantile_gamma_(\d+)_(\d+)_predictions\.npz$"
)


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denom = np.abs(actual) + np.abs(predicted)
    values = np.where(
        denom == 0,
        0.0,
        200.0 * np.abs(predicted - actual) / (denom + 1e-8),
    )
    return float(np.mean(values))


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down family-wise-error adjustment without an extra dependency."""
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running_max = 0.0
    m = len(values)
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * values[idx])
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def years(label_idx: np.ndarray) -> np.ndarray:
    idx = label_idx.astype(int)
    if idx.min() < 0 or idx.max() >= len(DATE_INDEX):
        raise ValueError("saved label_idx is outside the documented date index")
    return DATE_INDEX[idx].year.to_numpy()


def arrays(z: np.lib.npyio.NpzFile, split: str, h: int) -> dict[str, np.ndarray]:
    names = ("actual", "median", "lower", "upper", "persistence", "label_idx", "group")
    return {name: z[f"{split}_h{h}_{name}"] for name in names}


def validation_mask(store: dict[str, np.ndarray]) -> np.ndarray:
    mask = years(store["label_idx"]) == 2023
    if not mask.any():
        raise ValueError("artifact contains no 2023 validation labels")
    return mask


def fit_interval_scale(store: dict[str, np.ndarray]) -> float:
    mask = validation_mask(store)
    actual = store["actual"][mask]
    median = store["median"][mask]
    lo_dist = np.maximum(median - store["lower"][mask], 1e-8)
    hi_dist = np.maximum(store["upper"][mask] - median, 1e-8)
    required_scale = np.maximum(
        (median - actual) / lo_dist,
        (actual - median) / hi_dist,
    )
    return float(np.quantile(required_scale, 0.8))


def product_error_differences(store: dict[str, np.ndarray]) -> pd.Series:
    differential = (
        np.abs(store["actual"] - store["median"])
        - np.abs(store["actual"] - store["persistence"])
    )
    frame = pd.DataFrame(
        {"product": store["group"].astype(int), "differential": differential}
    )
    return frame.groupby("product", sort=True)["differential"].mean()


def build_champion_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    test_differences: dict[int, pd.Series] = {}

    with np.load(CHAMPION_NPZ) as z:
        for h in HORIZONS:
            val = arrays(z, "val", h)
            test = arrays(z, "test", h)
            scale = fit_interval_scale(val)
            cal_lower = test["median"] - scale * (test["median"] - test["lower"])
            cal_upper = test["median"] + scale * (test["upper"] - test["median"])
            covered = (test["actual"] >= cal_lower) & (test["actual"] <= cal_upper)
            product_diff = product_error_differences(test)
            test_differences[h] = product_diff
            test_years = years(test["label_idx"])

            row: dict[str, float] = {
                "Horizon": h,
                "Gamma": 4.5,
                "MAE": mae(test["actual"], test["median"]),
                "SMAPE": smape(test["actual"], test["median"]),
                "RMSE": rmse(test["actual"], test["median"]),
                "Baseline_MAE": mae(test["actual"], test["persistence"]),
                "Baseline_SMAPE": smape(test["actual"], test["persistence"]),
                "Baseline_RMSE": rmse(test["actual"], test["persistence"]),
                "CalPIScale": scale,
                "CalPIWidth": float(np.mean(cal_upper - cal_lower)),
                "CalPICoverage": float(np.mean(covered)),
                "ProductWinRate_MAE": float((product_diff < 0).mean()),
            }
            for year in (2024, 2025):
                mask = test_years == year
                row[f"MAE_{year}"] = mae(test["actual"][mask], test["median"][mask])
                row[f"Baseline_MAE_{year}"] = mae(
                    test["actual"][mask], test["persistence"][mask]
                )
            rows.append(row)

    champion = pd.DataFrame(rows)
    champion.to_csv(RESULTS_DIR / "champion_gamma45_metrics.csv", index=False)

    dm_rows: list[dict[str, float]] = []
    for h, differential in test_differences.items():
        t_stat, p_ttest = stats.ttest_1samp(differential.to_numpy(), 0.0)
        w_stat, p_wilcoxon = stats.wilcoxon(differential.to_numpy())
        dm_rows.append(
            {
                "horizon": h,
                "n_products": len(differential),
                "mean_diff_thb": float(differential.mean()),
                "share_negative": float((differential < 0).mean()),
                "t_stat": float(t_stat),
                "p_ttest": float(p_ttest),
                "wilcoxon_stat": float(w_stat),
                "p_wilcoxon": float(p_wilcoxon),
            }
        )

    overall = pd.concat(test_differences, axis=1).mean(axis=1)
    t_stat, p_ttest = stats.ttest_1samp(overall.to_numpy(), 0.0)
    w_stat, p_wilcoxon = stats.wilcoxon(overall.to_numpy())
    dm_rows.append(
        {
            "horizon": 0,
            "n_products": len(overall),
            "mean_diff_thb": float(overall.mean()),
            "share_negative": float((overall < 0).mean()),
            "t_stat": float(t_stat),
            "p_ttest": float(p_ttest),
            "wilcoxon_stat": float(w_stat),
            "p_wilcoxon": float(p_wilcoxon),
        }
    )
    dm = pd.DataFrame(dm_rows)
    horizon_mask = dm["horizon"].ne(0)
    dm.loc[horizon_mask, "p_ttest_holm"] = holm_adjust(
        dm.loc[horizon_mask, "p_ttest"].to_numpy()
    )
    dm.loc[horizon_mask, "p_wilcoxon_holm"] = holm_adjust(
        dm.loc[horizon_mask, "p_wilcoxon"].to_numpy()
    )
    dm.to_csv(RESULTS_DIR / "dm_tests.csv", index=False)
    return champion, dm


def build_validation_selection() -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for path in sorted(EXPERIMENTS_DIR.glob("tft_hw_quantile_gamma_*_predictions.npz")):
        match = PRIMARY_NPZ.match(path.name)
        if not match:
            continue
        gamma = float(f"{match.group(1)}.{match.group(2)}")
        row: dict[str, float | int] = {"Gamma": gamma}
        with np.load(path) as z:
            for h in HORIZONS:
                val = arrays(z, "val", h)
                mask = validation_mask(val)
                row[f"MAE_{h}"] = mae(val["actual"][mask], val["median"][mask])
                row[f"N_{h}"] = int(mask.sum())
        row["MAE_Overall"] = float(np.mean([row[f"MAE_{h}"] for h in HORIZONS]))
        rows.append(row)

    selection = pd.DataFrame(rows).sort_values("Gamma").reset_index(drop=True)
    found = tuple(selection["Gamma"].tolist())
    if found != EXPECTED_GAMMAS:
        raise RuntimeError(
            "the full primary 17-point gamma sweep is required; "
            f"expected {EXPECTED_GAMMAS}, found {found}"
        )
    selected = float(selection.loc[selection["MAE_Overall"].idxmin(), "Gamma"])
    if selected != 4.5:
        raise RuntimeError(f"validation selects gamma={selected}, not the reported gamma=4.5")
    selection.to_csv(RESULTS_DIR / "gamma_validation_summary.csv", index=False)
    return selection


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_provenance(selection: pd.DataFrame) -> None:
    payload = {
        "artifact": CHAMPION_NPZ.name,
        "artifact_sha256": sha256(CHAMPION_NPZ),
        "selection_split": "2023 labels",
        "test_split": "2024-2025 labels",
        "reported_horizons_business_days": list(HORIZONS),
        "candidate_gammas": selection["Gamma"].tolist(),
        "selection_metric": "mean MAE across reported horizons",
        "selected_gamma": 4.5,
        "sweep_seed": 42,
        "selected_artifact_seed": 42,
        "seed_sensitivity": [42, 43, 44],
        "interval_method": "validation-fitted conformal-style scaling; no formal coverage guarantee",
        "test_samples_per_horizon": 110292,
        "products": 404,
    }
    out = RESULTS_DIR / "publication_provenance.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not CHAMPION_NPZ.exists():
        raise FileNotFoundError(
            f"missing champion predictions: {CHAMPION_NPZ}\n"
            "Set CROP_EXPERIMENTS_DIR or regenerate the selected model first."
        )
    champion, dm = build_champion_tables()
    selection = build_validation_selection()
    write_provenance(selection)
    print(
        "wrote champion metrics, clustered tests, validation selection, and provenance\n"
        f"selected gamma: {selection.loc[selection['MAE_Overall'].idxmin(), 'Gamma']}\n"
        f"overall test MAE: model={champion.MAE.mean():.4f}, "
        f"persistence={champion.Baseline_MAE.mean():.4f}\n"
        f"overall product-level p-value: {dm.loc[dm.horizon == 0, 'p_ttest'].iloc[0]:.3g}"
    )


if __name__ == "__main__":
    main()
