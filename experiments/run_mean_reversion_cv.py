"""Rolling-origin selection of the mean-reversion design.

A single validation year is a fragile selector here, because 2023 contained an
unusual rice price spike. This script selects the design by rolling-origin
validation over three years instead, and is the last selection protocol run
in this study (the paper reports it together with the single-year protocols
of experiments/run_mean_reversion.py and experiments/run_mean_reversion_v2.py).

Protocol, fixed before any score is seen:
  for each validation year Y in 2021, 2022, 2023, fit every candidate on
  labels before 1 January Y and score every product-origin window whose label
  falls in Y (origins with at least 250 business days of history). The
  criterion is the geometric mean of the relative MAE over the three years
  and the four horizons. The best candidate is refitted on labels through
  2023 and scored once on the matched 2024-2025 test windows.

Candidate set: the 16 mean-reversion designs of run_mean_reversion_v2.py
(single, two-anchor, gap-dependent and both, pooled by group or family,
fitted to relative or THB error).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import itertools, json
import numpy as np
import pandas as pd

from paths import CHAMPION_NPZ, EXPERIMENTS_DIR, RESULTS_DIR
from experiments.run_mean_reversion import FIT_STRIDE, MIN_HISTORY, anchors, load
from experiments.run_mean_reversion_v2 import G, fit_cell, rel_mae, gmean
from src.data.tft_loader import build_dataset
from src.models.train_tft import time_idx_for_date

H = [20, 60, 120, 250]
YEARS = [2021, 2022, 2023]


def main():
    P, val_start, test_start, codes, group, family, level = load()
    full_df, _ = build_dataset()
    year_start = {y: time_idx_for_date(full_df, f"{y}-01-01") for y in YEARS + [2024]}
    A = anchors(P)
    m750, mexp = A["trail750"], A["expanding"]
    T = P.shape[1]
    pools = {"group": group, "family": family}
    n = len(codes)

    def gapsize(g, o):
        return np.abs(m750[g, o] - P[g, o]) / P[g, o]

    def fit(structure, cells, objective, h, label_end):
        o = np.arange(MIN_HISTORY, T - h, FIT_STRIDE); o = o[o + h < label_end]
        g = np.repeat(np.arange(n), len(o)); o = np.tile(o, n)
        r = P[g, o + h] - P[g, o]; g1 = m750[g, o] - P[g, o]; g2 = mexp[g, o] - P[g, o]
        w = np.ones(len(g)) if objective == "thb" else 1.0 / level[g]
        two, state = structure in ("two", "two_state"), structure in ("state", "two_state")
        prm = {}
        for c in np.unique(cells):
            k = cells[g] == c
            if state:
                gs = gapsize(g[k], o[k]); cut = float(np.median(gs)); idx = np.flatnonzero(k)
                prm[(c, "small")] = fit_cell(r[idx[gs <= cut]], g1[idx[gs <= cut]], g2[idx[gs <= cut]], w[idx[gs <= cut]], two)
                prm[(c, "large")] = fit_cell(r[idx[gs > cut]], g1[idx[gs > cut]], g2[idx[gs > cut]], w[idx[gs > cut]], two)
                prm[(c, "cut")] = cut
            else:
                prm[(c, "all")] = fit_cell(r[k], g1[k], g2[k], w[k], two)
        return prm

    def predict(structure, cells, prm, g, o):
        p = P[g, o]; g1 = m750[g, o] - p; g2 = mexp[g, o] - p
        state = structure in ("state", "two_state"); gs = gapsize(g, o)
        a = np.zeros(len(g)); b = np.zeros(len(g)); cg = cells[g]
        for c in np.unique(cg):
            k = cg == c
            if state:
                lg = k & (gs > prm[(c, "cut")]); sm = k & ~lg
                a[sm], b[sm] = prm[(c, "small")]; a[lg], b[lg] = prm[(c, "large")]
            else:
                a[k], b[k] = prm[(c, "all")]
        return p + a * g1 + b * g2

    rows = []
    for structure, pool, objective in itertools.product(["single", "two", "state", "two_state"], pools, ["relative", "thb"]):
        name = f"{structure}/{pool}/{objective}"; cells = pools[pool]
        rel = {}
        for y in YEARS:
            for h in H:
                prm = fit(structure, cells, objective, h, year_start[y])
                o = np.arange(MIN_HISTORY, T - h); l = o + h
                o = o[(l >= year_start[y]) & (l < year_start[y + 1] if y < 2023 else l < year_start[2024])]
                g = np.repeat(np.arange(n), len(o)); o = np.tile(o, n)
                f = predict(structure, cells, prm, g, o)
                rel[(y, h)] = rel_mae(P[g, o + h], f, P[g, o], g)
        score = gmean(list(rel.values()))
        rows.append(dict(candidate=name, cv_rel=score, **{f"rel_{y}": gmean([rel[(y, h)] for h in H]) for y in YEARS}))
        print(f"{name:28s} rolling-origin relMAE {score:.4f}  by year " +
              " ".join(f"{y}:{gmean([rel[(y, h)] for h in H]):.4f}" for y in YEARS), flush=True)

    cand = pd.DataFrame(rows).sort_values("cv_rel")
    cand.to_csv(RESULTS_DIR / "mean_reversion_cv_candidates.csv", index=False)
    sel = cand.iloc[0].candidate
    structure, pool, objective = sel.split("/")
    cells = pools[pool]
    print(f"\nselected by rolling-origin validation: {sel}")

    tft = np.load(CHAMPION_NPZ)
    rice = group == "Wholesale Rice & Bags"
    out, rel, thb, xr = {}, [], [], []
    for h in H:
        prm = fit(structure, cells, objective, h, test_start)
        g = tft[f"test_h{h}_group"].astype(int); l = tft[f"test_h{h}_label_idx"].astype(int); o = l - h
        f = predict(structure, cells, prm, g, o)
        out.update({f"test_h{h}_group": g, f"test_h{h}_label_idx": l, f"test_h{h}_actual": P[g, l],
                    f"test_h{h}_persistence": P[g, o], f"test_h{h}_median": f})
        # 2023 validation-window forecasts from the fit through 2022, so the evaluation harness can score it
        gv = tft[f"val_h{h}_group"].astype(int); lv = tft[f"val_h{h}_label_idx"].astype(int); kv = lv >= val_start
        gv, lv = gv[kv], lv[kv]; ov = lv - h
        fv = predict(structure, cells, fit(structure, cells, objective, h, val_start), gv, ov)
        out.update({f"val_h{h}_group": gv, f"val_h{h}_label_idx": lv, f"val_h{h}_actual": P[gv, lv],
                    f"val_h{h}_persistence": P[gv, ov], f"val_h{h}_median": fv})
        a = P[g, l]
        rel.append(rel_mae(a, f, P[g, o], g)); thb.append(np.mean(np.abs(a - f))); xr.append(np.mean(np.abs(a - f)[~rice[g]]))
        print(f"test h={h:3d}: MAE {thb[-1]:7.2f}  relMAE {rel[-1]:.4f}  MAE excl. rice {xr[-1]:6.2f}")
    print(f"TEST overall: MAE {np.mean(thb):.2f}  relMAE {gmean(rel):.4f}  MAE excl. rice {np.mean(xr):.2f}")
    np.savez_compressed(EXPERIMENTS_DIR / "variants" / "mean_reversion_cv_selected.npz", **out)
    json.dump({"selected": sel}, open(RESULTS_DIR / "mean_reversion_cv_selected.json", "w"), indent=1)


if __name__ == "__main__":
    main()
