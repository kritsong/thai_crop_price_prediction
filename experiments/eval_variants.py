"""Scale-aware scoring of TFT variants against persistence and damped persistence.

THB mean absolute error is dominated by the most expensive products (the 46
wholesale rice series cost about 35 times the median product), so it cannot
tell a broad improvement from a gain on one family. This script reports, per
horizon and on both the 2023 validation labels and the 2024-2025 test windows,

  MAE        THB mean absolute error over all windows
  MAE_xr     the same excluding the Wholesale Rice & Bags family
  RelMAE     geometric mean over products of MAE_model / MAE_persistence
             (AvgRelMAE, Davydenko and Fildes 2013)
  fam_wins   commodity families (group x category) whose mean product
             differential favors the model

Selection rule, fixed before any variant was scored: the lowest mean
validation RelMAE across the four reporting horizons.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse, glob, os
import numpy as np
import pandas as pd

from paths import CHAMPION_NPZ, EXPERIMENTS_DIR, RESULTS_DIR
from src.data.tft_loader import build_dataset
from src.models.train_tft import time_idx_for_date

H = [20, 60, 120, 250]
RICE = "Wholesale Rice & Bags"


def load_panel():
    full_df, _ = build_dataset()
    val_start = time_idx_for_date(full_df, "2023-01-02")
    # pytorch-forecasting encodes group ids in sorted order
    codes = np.unique(full_df["product_id"].astype(str))
    pos = {p: i for i, p in enumerate(codes)}
    T = int(full_df.time_idx.max()) + 1
    P = np.full((len(codes), T), np.nan)
    P[full_df.product_id.astype(str).map(pos).to_numpy(), full_df.time_idx.to_numpy()] = full_df.price_min.to_numpy()
    summ = pd.read_csv(RESULTS_DIR / "eda_product_summary.csv").set_index("product_id")
    grp_en = summ["group"].astype(str).reindex(codes).to_numpy()
    cat_en = summ["category"].astype(str).reindex(codes).to_numpy()
    family = np.char.add(np.char.add(grp_en.astype(str), " | "), cat_en.astype(str))
    return P, val_start, grp_en == RICE, family


def windows(z, split, h, val_start):
    g = z[f"{split}_h{h}_group"].astype(int)
    li = z[f"{split}_h{h}_label_idx"].astype(int)
    keep = li >= val_start if split == "val" else np.ones(len(li), bool)
    return {k: z[f"{split}_h{h}_{k}"][keep] for k in ["actual", "median", "persistence"]} | {"g": g[keep], "li": li[keep]}


def score(actual, pred, persist, g, rice, family):
    e, ep = np.abs(actual - pred), np.abs(actual - persist)
    per = pd.DataFrame({"g": g, "e": e, "ep": ep}).groupby("g").mean()
    ok = per.ep > 0
    rel = float(np.exp(np.mean(np.log(np.maximum(per.e[ok], 1e-9) / per.ep[ok]))))
    xr = ~rice[g]
    diff = (per.e - per.ep)
    fam = diff.groupby(family[diff.index]).mean()
    return dict(MAE=float(e.mean()), MAE_xr=float(e[xr].mean()), RelMAE=rel,
                fam_wins=int((fam < 0).sum()), n_fam=len(fam))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default=str(EXPERIMENTS_DIR / "variants" / "*.npz"))
    args = ap.parse_args()

    P, val_start, rice, family = load_panel()
    C = np.cumsum(P, axis=1)
    N = np.arange(1, P.shape[1] + 1)

    sources = {"hwtft_published": str(CHAMPION_NPZ)}
    for f in sorted(glob.glob(args.pattern)):
        if "_smoke" not in f:
            sources[os.path.splitext(os.path.basename(f))[0]] = f

    rows = []
    ref = np.load(CHAMPION_NPZ)
    grid = np.linspace(0, 1, 101)
    for h in H:
        # reference forecasts on the published windows (identical for every variant)
        v = windows(ref, "val", h, val_start)
        mu_v = C[v["g"], v["li"] - h] / N[v["li"] - h]
        lam = grid[int(np.argmin([np.mean(np.abs(v["actual"] - (v["persistence"] + l * (mu_v - v["persistence"])))) for l in grid]))]
        for split in ["val", "test"]:
            w = windows(ref, split, h, val_start)
            mu = C[w["g"], w["li"] - h] / N[w["li"] - h]
            for name, pred in [("persistence", w["persistence"]),
                               ("damped_persistence", w["persistence"] + lam * (mu - w["persistence"]))]:
                rows.append(dict(model=name, split=split, h=h,
                                 **score(w["actual"], pred, w["persistence"], w["g"], rice, family)))
    for name, path in sources.items():
        z = np.load(path)
        for h in H:
            for split in ["val", "test"]:
                w = windows(z, split, h, val_start)
                rows.append(dict(model=name, split=split, h=h,
                                 **score(w["actual"], w["median"], w["persistence"], w["g"], rice, family)))

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "variant_scores.csv", index=False)

    sel = (df[df.split == "val"].groupby("model")["RelMAE"].apply(lambda s: float(np.exp(np.mean(np.log(s)))))
           .rename("val_RelMAE").sort_values())
    print("validation ranking (mean RelMAE over horizons, geometric):")
    print(sel.round(4).to_string())
    learned = [m for m in sel.index if m not in ("persistence", "damped_persistence")]
    print(f"\nselected by validation among learned variants: {learned[0]}")

    for split in ["val", "test"]:
        print(f"\n== {split}: MAE / MAE excl. rice / RelMAE / family wins")
        t = df[df.split == split]
        for m in sel.index:
            cells = []
            for h in H:
                r = t[(t.model == m) & (t.h == h)].iloc[0]
                cells.append(f"{r.MAE:7.2f}/{r.MAE_xr:6.2f}/{r.RelMAE:5.3f}/{r.fam_wins:2d}")
            o = t[t.model == m]
            cells.append(f"{o.MAE.mean():7.2f}/{o.MAE_xr.mean():6.2f}/{np.exp(np.log(o.RelMAE).mean()):5.3f}")
            print(f"{m:34s} " + "  ".join(cells))


if __name__ == "__main__":
    main()
