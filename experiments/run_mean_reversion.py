"""Horizon-wise mean-reversion forecaster.

    y_hat(t+h) = p(t) + lambda_{h,c} * (mu(t) - p(t))

p(t) is the last observed price at the forecast origin, mu(t) a long-run price
level computed only from data up to the origin, and lambda_{h,c} in [0, 1] the
fraction of the gap expected to close within h business days for pooling cell
c. lambda = 0 is persistence and lambda = 1 is full reversion.

Protocol, fixed before any candidate was scored:
  1. fit lambda for every candidate on labels through 2022
  2. select one candidate on the 2023 validation labels by the lowest mean
     relative MAE (geometric mean over products of MAE / persistence MAE,
     then geometric mean over the four horizons)
  3. refit the selected candidate on labels through 2023 and score the
     2024-2025 test windows once

Candidates: anchor mu in {expanding mean, trailing 500-day, trailing 750-day},
lambda pooled {globally, by product group, by commodity family}, fitted to
minimize {THB absolute error, relative absolute error}. Evaluation windows are
the matched windows of the TFT artifact, so every row is comparable with Table 5.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import itertools, json
import numpy as np
import pandas as pd

from paths import CHAMPION_NPZ, RESULTS_DIR
from src.data.tft_loader import build_dataset
from src.models.train_tft import time_idx_for_date

H = [20, 60, 120, 250]
GRID = np.round(np.linspace(0, 1, 101), 2)
FIT_STRIDE = 5          # fitting origins are subsampled; consecutive origins overlap almost completely
MIN_HISTORY = 250       # origins need a year of history so every anchor is defined


def load():
    full_df, _ = build_dataset()
    val_start = time_idx_for_date(full_df, "2023-01-02")
    test_start = time_idx_for_date(full_df, "2024-01-01")
    codes = np.unique(full_df["product_id"].astype(str))   # pytorch-forecasting group-id order
    pos = {p: i for i, p in enumerate(codes)}
    T = int(full_df.time_idx.max()) + 1
    P = np.full((len(codes), T), np.nan)
    P[full_df.product_id.astype(str).map(pos).to_numpy(), full_df.time_idx.to_numpy()] = full_df.price_min.to_numpy()
    assert np.isfinite(P).all()
    summ = pd.read_csv(RESULTS_DIR / "eda_product_summary.csv").set_index("product_id")
    group = summ["group"].astype(str).reindex(codes).to_numpy()
    family = (summ["group"].astype(str) + " | " + summ["category"].astype(str)).reindex(codes).to_numpy()
    level = np.array([P[i, :val_start].mean() for i in range(len(codes))])   # 2018-2022 mean price
    return P, val_start, test_start, codes, group, family, level


def anchors(P):
    T = P.shape[1]
    C = np.concatenate([np.zeros((P.shape[0], 1)), np.cumsum(P, axis=1)], axis=1)

    def trailing(w):
        out = np.full_like(P, np.nan)
        idx = np.arange(T)
        lo = np.maximum(idx + 1 - w, 0)
        out[:, :] = (C[:, idx + 1] - C[:, lo]) / (idx + 1 - lo)
        return out

    return {"expanding": C[:, 1:] / np.arange(1, T + 1), "trail500": trailing(500), "trail750": trailing(750)}


def fit_lambda(a, p, mu, cell, weight):
    """Per pooling cell, the grid value of lambda minimizing weighted absolute error."""
    lam = {}
    gap = mu - p
    for c in np.unique(cell):
        k = cell == c
        errs = [np.sum(weight[k] * np.abs(a[k] - (p[k] + g * gap[k]))) for g in GRID]
        lam[c] = float(GRID[int(np.argmin(errs))])
    return lam


def rel_mae(a, f, p, g):
    per = pd.DataFrame({"g": g, "e": np.abs(a - f), "ep": np.abs(a - p)}).groupby("g").mean()
    per = per[per.ep > 0]
    return float(np.exp(np.mean(np.log(np.maximum(per.e, 1e-12) / per.ep))))


def main():
    P, val_start, test_start, codes, group, family, level = load()
    A = anchors(P)
    z = np.load(CHAMPION_NPZ)
    T = P.shape[1]
    pools = {"global": np.zeros(len(codes), dtype=object), "group": group, "family": family}

    def fit_rows(h, label_end):
        o = np.arange(MIN_HISTORY, T - h, FIT_STRIDE)
        o = o[o + h < label_end]
        g = np.repeat(np.arange(len(codes)), len(o))
        oo = np.tile(o, len(codes))
        return g, oo, oo + h

    def eval_rows(split, h):
        g = z[f"{split}_h{h}_group"].astype(int)
        li = z[f"{split}_h{h}_label_idx"].astype(int)
        keep = li >= val_start if split == "val" else np.ones(len(li), bool)
        return g[keep], li[keep] - h, li[keep]

    rows, lambdas = [], {}
    for anchor, pool, objective in itertools.product(A, pools, ["thb", "relative"]):
        name = f"{anchor}/{pool}/{objective}"
        mu_all, cells = A[anchor], pools[pool]
        val_rel, lam_h = [], {}
        for h in H:
            g, o, l = fit_rows(h, val_start)
            w = np.ones(len(g)) if objective == "thb" else 1.0 / level[g]
            lam = fit_lambda(P[g, l], P[g, o], mu_all[g, o], cells[g], w)
            lam_h[h] = lam
            g, o, l = eval_rows("val", h)
            f = P[g, o] + np.array([lam[c] for c in cells[g]]) * (mu_all[g, o] - P[g, o])
            val_rel.append(rel_mae(P[g, l], f, P[g, o], g))
        score = float(np.exp(np.mean(np.log(val_rel))))
        rows.append(dict(candidate=name, anchor=anchor, pool=pool, objective=objective,
                         **{f"val_rel_{h}": r for h, r in zip(H, val_rel)}, val_rel_mean=score))
        lambdas[name] = lam_h
        print(f"{name:32s} validation relMAE by horizon {np.round(val_rel, 4).tolist()}  mean {score:.4f}", flush=True)

    cand = pd.DataFrame(rows).sort_values("val_rel_mean")
    cand.to_csv(RESULTS_DIR / "mean_reversion_candidates.csv", index=False)
    best = cand.iloc[0]
    print(f"\nselected on validation: {best.candidate} (mean relMAE {best.val_rel_mean:.4f})")

    # refit the selected design on every label through 2023, then score the test windows once
    mu_all, cells = A[best.anchor], pools[best.pool]
    final_lam, out = {}, {}
    for h in H:
        g, o, l = fit_rows(h, test_start)
        w = np.ones(len(g)) if best.objective == "thb" else 1.0 / level[g]
        final_lam[h] = fit_lambda(P[g, l], P[g, o], mu_all[g, o], cells[g], w)
        g, o, l = eval_rows("test", h)
        lam_row = np.array([final_lam[h][c] for c in cells[g]])
        out[f"test_h{h}_group"] = g
        out[f"test_h{h}_label_idx"] = l
        out[f"test_h{h}_actual"] = P[g, l]
        out[f"test_h{h}_persistence"] = P[g, o]
        out[f"test_h{h}_median"] = P[g, o] + lam_row * (mu_all[g, o] - P[g, o])
        out[f"test_h{h}_lambda"] = lam_row
    # validation forecasts of the selected design (lambda fitted through 2022), for the evaluation harness
    for h in H:
        g, o, l = eval_rows("val", h)
        lam_row = np.array([lambdas[best.candidate][h][c] for c in cells[g]])
        out[f"val_h{h}_group"] = g
        out[f"val_h{h}_label_idx"] = l
        out[f"val_h{h}_actual"] = P[g, l]
        out[f"val_h{h}_persistence"] = P[g, o]
        out[f"val_h{h}_median"] = P[g, o] + lam_row * (mu_all[g, o] - P[g, o])

    from paths import EXPERIMENTS_DIR
    (EXPERIMENTS_DIR / "variants").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EXPERIMENTS_DIR / "variants" / "mean_reversion_selected.npz", **out)
    json.dump({"selected": best.candidate,
               "lambda_fit_through_2023": {str(h): final_lam[h] for h in H},
               "lambda_fit_through_2022": {str(h): lambdas[best.candidate][h] for h in H}},
              open(RESULTS_DIR / "mean_reversion_selected.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("saved selected forecasts and lambdas")


if __name__ == "__main__":
    main()
