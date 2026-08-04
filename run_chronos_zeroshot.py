"""Chronos-Bolt-Base, zero-shot, on the same test windows as everything else.

Each window gets the product's price history up to the origin (capped at the
model's context limit) and has to predict 250 steps out. We score the median at
20, 60, 120 and 250 against the same actuals as every other model. No
fine-tuning, that is the whole point of including it.

Persistence is recomputed here as a check that the windows line up.
"""
import os, sys, io, json, glob, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

from src.data.loader import CropDataLoader, EmptySeriesError
from src.features.generator import FeatureGenerator
from chronos import BaseChronosPipeline

from paths import DATA_GLOB, RESULTS_DIR as _RESULTS_DIR

RESULTS_DIR = str(_RESULTS_DIR)
UNIVERSE = set(pd.read_csv(os.path.join(RESULTS_DIR, "metrics_by_horizon.csv"))["product_id"].unique())
H = [20, 60, 120, 250]
CONTEXT = 512
PRED_LEN = 250
BATCH = 256


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

    y = df_aligned["price_min"].to_numpy(np.float32)
    pos_of_date = {d: i for i, d in enumerate(df_aligned["date"].to_numpy())}
    wins = []
    for _, row in test_df.iterrows():
        o = pos_of_date[row["date"]] - 1
        wins.append({
            "pid": pid, "origin": o,
            "actuals": {h: float(row[f"target_{h}"]) for h in H},
            "persistence": float(y[o]),
            "context": y[max(0, o + 1 - CONTEXT):o + 1].copy(),
        })
    return wins


def main():
    t0 = time.time()
    filepaths = glob.glob(DATA_GLOB)
    parts = Parallel(n_jobs=-1)(delayed(one_product)(fp) for fp in filepaths)
    wins = [w for p in parts if p for w in p]
    print(f"windows: {len(wins):,}  products: {len(set(w['pid'] for w in wins))}  "
          f"({time.time()-t0:.0f}s to assemble)")

    # sanity on persistence before any model runs
    for h in H:
        mae = np.mean([abs(w["actuals"][h] - w["persistence"]) for w in wins])
        print(f"persistence h{h}: {mae:.2f}")

    pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-base", device_map="cuda", torch_dtype=torch.bfloat16)
    print("pipeline loaded")

    q_idx = {20: 19, 60: 59, 120: 119, 250: 249}
    preds = {h: np.empty(len(wins), dtype=np.float64) for h in H}
    n_batches = (len(wins) + BATCH - 1) // BATCH
    for bi in range(n_batches):
        chunk = wins[bi * BATCH:(bi + 1) * BATCH]
        ctx = [torch.tensor(w["context"]) for w in chunk]
        with torch.no_grad():
            quantiles, _ = pipe.predict_quantiles(
                ctx, prediction_length=PRED_LEN, quantile_levels=[0.5])
        med = quantiles[:, :, 0].float().cpu().numpy()   # (batch, 250)
        for h in H:
            preds[h][bi * BATCH:bi * BATCH + len(chunk)] = med[:, q_idx[h]]
        if bi % 50 == 0:
            print(f"batch {bi+1}/{n_batches}  elapsed {time.time()-t0:.0f}s")

    rows = {"model": "chronos_bolt_base_zeroshot"}
    for h in H:
        act = np.array([w["actuals"][h] for w in wins])
        f = preds[h]
        rows[f"mae_{h}"] = float(np.mean(np.abs(act - f)))
        rows[f"smape_{h}"] = smape(act, f)
        print(f"chronos h{h}: MAE {rows[f'mae_{h}']:.2f}  SMAPE {rows[f'smape_{h}']:.2f}%")
    rows["mae_overall"] = float(np.mean([rows[f"mae_{h}"] for h in H]))
    rows["smape_overall"] = float(np.mean([rows[f"smape_{h}"] for h in H]))
    print(f"chronos overall MAE: {rows['mae_overall']:.2f}")

    pd.DataFrame([rows]).to_csv(os.path.join(RESULTS_DIR, "chronos_zeroshot.csv"), index=False)
    print(f"saved results/chronos_zeroshot.csv  ({(time.time()-t0)/60:.1f} min total)")


if __name__ == "__main__":
    main()
