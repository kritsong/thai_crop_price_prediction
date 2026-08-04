"""Seasonal-naive and drift baselines, on the same test windows as everything else.

Both are parameter-free and both are decent at long horizons, which is exactly
where we claim an advantage, so they are the honest ones to beat.

Seasonal naive uses a 250-business-day period, i.e. the price one year earlier.
Drift extrapolates the line through the first and last observed points.

The script also recomputes persistence and asserts it matches the published
numbers, which is a cheap way to prove the windows really are identical.
"""
import os, sys, io, json, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.data.loader import CropDataLoader, EmptySeriesError
from src.features.generator import FeatureGenerator

from paths import DATA_GLOB, RESULTS_DIR as _RESULTS_DIR

RESULTS_DIR = str(_RESULTS_DIR)
UNIVERSE = set(pd.read_csv(os.path.join(RESULTS_DIR, "metrics_by_horizon.csv"))["product_id"].unique())
H = [20, 60, 120, 250]
SEASON = 250


def smape(a, f):
    denom = (np.abs(a) + np.abs(f)) / 2.0
    return float(np.mean(np.where(denom == 0, 0.0, 100.0 * np.abs(f - a) / denom)))


def one_product(fp):
    pid = os.path.basename(fp).split(".")[0]
    if pid not in UNIVERSE:
        return None
    loader = CropDataLoader()
    gen = FeatureGenerator()
    try:
        df_aligned = loader.load_and_preprocess(fp)
        if len(df_aligned) < 70:
            return None
        df_feat = gen.generate_features(df_aligned)
        if len(df_feat) == 0:
            return None
        _, test_df = gen.split_train_test(df_feat)
        if len(test_df) == 0:
            return None
    except (json.JSONDecodeError, EmptySeriesError, KeyError, Exception):
        return None

    y = df_aligned["price_min"].to_numpy(float)
    pos_of_date = {d: i for i, d in enumerate(df_aligned["date"].to_numpy())}

    recs = []
    for _, row in test_df.iterrows():
        D = row["date"]
        p = pos_of_date[D]          # row date position; origin is p-1
        o = p - 1
        for h in H:
            actual = row[f"target_{h}"]           # y at position p + h - 1 = o + h
            lbl = o + h
            recs.append({
                "pid": pid, "h": h, "actual": actual,
                "persistence": y[o],
                "snaive": y[lbl - SEASON],
                "drift": y[o] + h * (y[o] - y[0]) / o,
            })
    return recs


def main():
    filepaths = glob.glob(DATA_GLOB)
    parts = Parallel(n_jobs=-1)(delayed(one_product)(fp) for fp in filepaths)
    recs = [r for p in parts if p for r in p]
    df = pd.DataFrame(recs)
    print(f"products: {df.pid.nunique()}  windows/horizon: {len(df)//len(H):,}")
    assert df.pid.nunique() == len(UNIVERSE)

    rows = []
    for model in ["persistence", "snaive", "drift"]:
        out = {"model": model}
        for h in H:
            sub = df[df.h == h]
            out[f"mae_{h}"] = float(np.mean(np.abs(sub.actual - sub[model])))
            out[f"smape_{h}"] = smape(sub.actual.values, sub[model].values)
        out["mae_overall"] = float(np.mean([out[f"mae_{h}"] for h in H]))
        out["smape_overall"] = float(np.mean([out[f"smape_{h}"] for h in H]))
        rows.append(out)
        print(f"{model:12s} " + "  ".join(f"h{h}={out[f'mae_{h}']:7.2f}" for h in H)
              + f"  overall={out['mae_overall']:7.2f}")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RESULTS_DIR, "extra_baselines.csv"), index=False)
    print("saved results/extra_baselines.csv")

    pub = {20: 15.109673, 60: 30.859187, 120: 51.026358, 250: 83.840534}
    mine = res[res.model == "persistence"].iloc[0]
    for h in H:
        assert abs(mine[f"mae_{h}"] - pub[h]) < 0.01, f"persistence mismatch at h={h}"
    print("SANITY PASSED: persistence reproduces Table 5 exactly")


if __name__ == "__main__":
    main()
