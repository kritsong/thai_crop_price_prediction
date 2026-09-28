"""Extended mean-reversion forecasters and a combination with the horizon-weighted TFT.

This candidate set was defined after the first test evaluation of the base
design in experiments/run_mean_reversion.py, and the paper discloses that. To
keep the test period out of every decision, the protocol below is fixed before
any extended candidate is scored and is identical to the base protocol.

  1. fit every mean-reversion candidate on labels through 2022
  2. score every candidate on the 2023 validation labels; the primary
     criterion is the mean relative MAE (geometric mean over products of
     MAE / persistence MAE, then over the four horizons), THB MAE is reported
  3. combination weights with the TFT can only be fitted on 2023 (the TFT
     was trained through 2022), so their validation score is cross-fitted:
     weights fitted on labels in January-June are scored on July-December
     and vice versa
  4. select the lowest validation criterion among all candidates, refit it on
     labels through 2023 (combination weights on all of 2023), and score the
     test windows once

Structures (each with family or group pooling, THB or relative objective)
  single     p + l (mu750 - p)                        the base design
  two        p + a (mu750 - p) + b (muexp - p)         two memories, a, b >= 0, a + b <= 1
  state      p + l_k (mu750 - p), k = small or large gap relative to the price
  two_state  both of the above
Combination: w_h * mean_reversion + (1 - w_h) * horizon-weighted TFT.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import itertools, json
import numpy as np
import pandas as pd

from paths import CHAMPION_NPZ, EXPERIMENTS_DIR, RESULTS_DIR
from experiments.run_mean_reversion import FIT_STRIDE, MIN_HISTORY, anchors, load

H = [20, 60, 120, 250]
G = np.round(np.linspace(0, 1, 101), 2)          # one-anchor grid, identical to the base design
G2 = np.round(np.arange(0, 1 + 1e-9, 0.02), 2)   # coarser grid for the two-anchor search


def fit_cell(r, g1, g2, w, two):
    """Grid LAD fit of r ~ a g1 + b g2 (b = 0 unless two), a, b >= 0, a + b <= 1."""
    if not two:
        e = [np.sum(w * np.abs(r - a * g1)) for a in G]
        return float(G[int(np.argmin(e))]), 0.0
    best = (np.inf, 0.0, 0.0)
    for a in G2:
        bs = G2[G2 <= 1 - a + 1e-9]
        e = np.abs(r[None, :] - a * g1[None, :] - bs[:, None] * g2[None, :]) @ w
        k = int(np.argmin(e))
        if e[k] < best[0]:
            best = (float(e[k]), float(a), float(bs[k]))
    return best[1], best[2]


def rel_mae(a, f, p, g):
    per = pd.DataFrame({"g": g, "e": np.abs(a - f), "ep": np.abs(a - p)}).groupby("g").mean()
    per = per[per.ep > 0]
    return float(np.exp(np.mean(np.log(np.maximum(per.e, 1e-12) / per.ep))))


def gmean(x):
    return float(np.exp(np.mean(np.log(x))))


def main():
    P, val_start, test_start, codes, group, family, level = load()
    A = anchors(P)
    m750, mexp = A["trail750"], A["expanding"]
    T = P.shape[1]
    pools = {"group": group, "family": family}
    tft = np.load(CHAMPION_NPZ)

    def fit_rows(h, label_end):
        o = np.arange(MIN_HISTORY, T - h, FIT_STRIDE); o = o[o + h < label_end]
        return np.repeat(np.arange(len(codes)), len(o)), np.tile(o, len(codes))

    def eval_rows(split, h):
        g = tft[f"{split}_h{h}_group"].astype(int); li = tft[f"{split}_h{h}_label_idx"].astype(int)
        keep = li >= val_start if split == "val" else np.ones(len(li), bool)
        return g[keep], li[keep] - h, li[keep], tft[f"{split}_h{h}_median"][keep]

    def gapsize(g, o):
        return np.abs(m750[g, o] - P[g, o]) / P[g, o]

    def fit(structure, cells, objective, h, label_end):
        g, o = fit_rows(h, label_end)
        r = P[g, o + h] - P[g, o]; g1 = m750[g, o] - P[g, o]; g2 = mexp[g, o] - P[g, o]
        w = np.ones(len(g)) if objective == "thb" else 1.0 / level[g]
        two = structure in ("two", "two_state")
        state = structure in ("state", "two_state")
        params = {}
        for c in np.unique(cells):
            k = cells[g] == c
            if state:
                # split at the cell's median relative gap on the fitting rows
                gs = gapsize(g[k], o[k]); cut = float(np.median(gs))
                for s, ks in [("small", gs <= cut), ("large", gs > cut)]:
                    idx = np.flatnonzero(k)[ks]
                    params[(c, s)] = fit_cell(r[idx], g1[idx], g2[idx], w[idx], two)
                params[(c, "cut")] = cut
            else:
                params[(c, "all")] = fit_cell(r[k], g1[k], g2[k], w[k], two)
        return params

    def predict(structure, cells, params, g, o):
        p = P[g, o]; g1 = m750[g, o] - p; g2 = mexp[g, o] - p
        state = structure in ("state", "two_state")
        gs = gapsize(g, o)
        a = np.zeros(len(g)); b = np.zeros(len(g))
        cg = cells[g]
        for c in np.unique(cg):
            k = cg == c
            if state:
                large = k & (gs > params[(c, "cut")])
                small = k & ~large
                a[small], b[small] = params[(c, "small")]
                a[large], b[large] = params[(c, "large")]
            else:
                a[k], b[k] = params[(c, "all")]
        return p + a * g1 + b * g2

    rows, store = [], {}
    for structure, pool, objective in itertools.product(["single", "two", "state", "two_state"], pools, ["relative", "thb"]):
        name = f"{structure}/{pool}/{objective}"
        cells = pools[pool]
        rel, thb, fits, vpred = [], [], {}, {}
        for h in H:
            prm = fit(structure, cells, objective, h, val_start)
            g, o, l, _ = eval_rows("val", h)
            f = predict(structure, cells, prm, g, o)
            rel.append(rel_mae(P[g, l], f, P[g, o], g)); thb.append(float(np.mean(np.abs(P[g, l] - f))))
            fits[h] = prm; vpred[h] = f
        store[name] = (structure, pool, objective, vpred)
        rows.append(dict(candidate=name, kind="mean_reversion", val_rel=gmean(rel), val_thb=float(np.mean(thb)),
                         **{f"val_rel_{h}": v for h, v in zip(H, rel)}))
        print(f"{name:28s} val relMAE {gmean(rel):.4f}  val THB MAE {np.mean(thb):6.2f}", flush=True)

    # combination with the TFT, cross-fitted inside 2023, for the best mean-reversion candidate
    cand = pd.DataFrame(rows).sort_values("val_rel")
    base = cand.iloc[0].candidate
    structure, pool, objective, vpred = store[base]
    from src.models.train_tft import time_idx_for_date
    from src.data.tft_loader import build_dataset
    full_df, _ = build_dataset()
    mid = time_idx_for_date(full_df, "2023-07-01")
    for combo in ["combo_global", "combo_family"]:
        rel, thb = [], []
        for h in H:
            g, o, l, ft = eval_rows("val", h)
            fm, a, p = vpred[h], P[g, l], P[g, o]
            cells = np.zeros(len(g), dtype=object) if combo == "combo_global" else family[g]
            out = np.empty(len(g))
            for fit_half, score_half in [(l < mid, l >= mid), (l >= mid, l < mid)]:
                for c in np.unique(cells):
                    kf = fit_half & (cells == c); ks = score_half & (cells == c)
                    wgt = np.ones(kf.sum()) if objective == "thb" else 1.0 / level[g[kf]]
                    errs = [np.sum(wgt * np.abs(a[kf] - (x * fm[kf] + (1 - x) * ft[kf])) ) for x in G]
                    x = G[int(np.argmin(errs))]
                    out[ks] = x * fm[ks] + (1 - x) * ft[ks]
            rel.append(rel_mae(a, out, p, g)); thb.append(float(np.mean(np.abs(a - out))))
        rows.append(dict(candidate=f"{combo}({base})", kind="combination", val_rel=gmean(rel), val_thb=float(np.mean(thb)),
                         **{f"val_rel_{h}": v for h, v in zip(H, rel)}))
        print(f"{combo}({base}) cross-fitted val relMAE {gmean(rel):.4f}  val THB MAE {np.mean(thb):6.2f}", flush=True)

    cand = pd.DataFrame(rows).sort_values("val_rel")
    cand.to_csv(RESULTS_DIR / "mean_reversion_v2_candidates.csv", index=False)
    sel = cand.iloc[0]
    print(f"\nselected on validation: {sel.candidate}  (val relMAE {sel.val_rel:.4f}, val THB {sel.val_thb:.2f})")

    # final fit and the single test evaluation
    is_combo = sel.kind == "combination"
    base_name = sel.candidate.split("(")[1].rstrip(")") if is_combo else sel.candidate
    structure, pool, objective, vpred = store[base_name]
    cells = pools[pool]
    out = {}
    for h in H:
        prm = fit(structure, cells, objective, h, test_start)
        g, o, l, ft = eval_rows("test", h)
        fm = predict(structure, cells, prm, g, o)
        f = fm
        if is_combo:
            gv, ov, lv, ftv = eval_rows("val", h)
            cv = np.zeros(len(gv), dtype=object) if sel.candidate.startswith("combo_global") else family[gv]
            ct = np.zeros(len(g), dtype=object) if sel.candidate.startswith("combo_global") else family[g]
            f = np.empty(len(g))
            for c in np.unique(ct):
                kv = cv == c; kt = ct == c
                wgt = np.ones(kv.sum()) if objective == "thb" else 1.0 / level[gv[kv]]
                errs = [np.sum(wgt * np.abs(P[gv[kv], lv[kv]] - (x * vpred[h][kv] + (1 - x) * ftv[kv]))) for x in G]
                x = G[int(np.argmin(errs))]
                f[kt] = x * fm[kt] + (1 - x) * ft[kt]
        out.update({f"test_h{h}_group": g, f"test_h{h}_label_idx": l, f"test_h{h}_actual": P[g, l],
                    f"test_h{h}_persistence": P[g, o], f"test_h{h}_median": f})
        gv, ov, lv, _ = eval_rows("val", h)
        out.update({f"val_h{h}_group": gv, f"val_h{h}_label_idx": lv, f"val_h{h}_actual": P[gv, lv],
                    f"val_h{h}_persistence": P[gv, ov], f"val_h{h}_median": vpred[h]})
    (EXPERIMENTS_DIR / "variants").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EXPERIMENTS_DIR / "variants" / "mean_reversion_v2_selected.npz", **out)
    json.dump({"selected": sel.candidate}, open(RESULTS_DIR / "mean_reversion_v2_selected.json", "w"), indent=1)

    rel, thb, xr = [], [], []
    rice = group == "Wholesale Rice & Bags"
    for h in H:
        a, f, p, g = out[f"test_h{h}_actual"], out[f"test_h{h}_median"], out[f"test_h{h}_persistence"], out[f"test_h{h}_group"]
        rel.append(rel_mae(a, f, p, g)); thb.append(np.mean(np.abs(a - f))); xr.append(np.mean(np.abs(a - f)[~rice[g]]))
        print(f"test h={h:3d}: MAE {thb[-1]:7.2f}  relMAE {rel[-1]:.4f}  MAE excl. rice {xr[-1]:6.2f}")
    print(f"TEST overall: MAE {np.mean(thb):.2f}  relMAE {gmean(rel):.4f}  MAE excl. rice {np.mean(xr):.2f}")


if __name__ == "__main__":
    main()
