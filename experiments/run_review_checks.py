"""Checks requested in peer review, on the same matched windows as every table.

  mr_sensitivity.csv      trailing-mean window (250, 500, 750, 1,000 business days) and
                          deviation-state quantile (0.4, 0.5, 0.6) of the selected design,
                          with the rolling-origin validation criterion and the test scores
  mr_fit_through_2022.csv the selected design fitted on labels through 2022 only, the
                          training window of the TFT, scored on the test windows
  mr_baselines.csv        statistical baselines with the same inductive bias, fitted on labels
                          through 2023 like the other references: one global shrinkage share
                          per horizon toward the trailing mean, simple exponential smoothing,
                          and damped Holt (additive damped trend), both per product
  mr_effect_sizes.csv     median and interquartile range of the product relative-error
                          differential, share of products improved, and a family bootstrap
                          interval for RelMAE

The selected design (two levels, two deviation states, group pooling, THB
objective) is fixed. The sensitivity grid is reported for interpretation and
does not change the selection.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import warnings
import numpy as np
import pandas as pd

from paths import CHAMPION_NPZ, EXPERIMENTS_DIR, RESULTS_DIR
from experiments.run_mean_reversion import MIN_HISTORY, load
from experiments.run_mean_reversion_v2 import rel_mae, gmean
from src.data.tft_loader import build_dataset
from src.models.train_tft import time_idx_for_date
from src.models.mean_reversion import MeanReversion

H = [20, 60, 120, 250]
YEARS = [2021, 2022, 2023]
RICE = "Wholesale Rice & Bags"
rng = np.random.default_rng(0)


def scores(P, g, o, h, f, rice):
    a = P[g, o + h]; p = P[g, o]
    xr = ~rice[g]
    return dict(RelMAE=rel_mae(a, f, p, g), RelMAExr=rel_mae(a[xr], f[xr], p[xr], g[xr]), MAE=float(np.mean(np.abs(a - f))), MAExr=float(np.mean(np.abs(a - f)[xr])),
                RMSE=float(np.sqrt(np.mean((a - f) ** 2))), MAPE=float(np.mean(100 * np.abs(a - f) / a)),
                SMAPE=float(np.mean(100 * np.abs(a - f) / ((np.abs(a) + np.abs(f)) / 2))))


def overall(per_h):
    out = {}
    for k in per_h[0]:
        v = [d[k] for d in per_h]
        out[f"{k}_overall"] = gmean(v) if k.startswith("RelMAE") else float(np.mean(v))
        out.update({f"{k}_{h}": x for h, x in zip(H, v)})
    return out


def main():
    P, val_start, test_start, codes, group, family, level = load()
    full_df, _ = build_dataset()
    ystart = {y: time_idx_for_date(full_df, f"{y}-01-01") for y in YEARS + [2024]}
    n, T = P.shape
    rice = group == RICE
    z = np.load(CHAMPION_NPZ)
    TW = {h: (z[f"test_h{h}_group"].astype(int), z[f"test_h{h}_label_idx"].astype(int) - h) for h in H}

    def cv_and_test(window=750, cut_q=0.5, structure="two_state", cells=group, label_end=test_start):
        m = MeanReversion(P, level, cells, structure, "thb", window=window, cut_q=cut_q)
        rel = []
        for y in YEARS:
            for h in H:
                m.fit(h, ystart[y])
                o = np.arange(MIN_HISTORY, T - h); l = o + h
                o = o[(l >= ystart[y]) & (l < ystart[y + 1])]
                g = np.repeat(np.arange(n), len(o)); o = np.tile(o, n)
                rel.append(rel_mae(P[g, o + h], m.predict(h, g, o), P[g, o], g))
        per_h, preds = [], {}
        for h in H:
            m.fit(h, label_end)
            g, o = TW[h]; f = m.predict(h, g, o); preds[h] = f
            per_h.append(scores(P, g, o, h, f, rice))
        return gmean(rel), overall(per_h), preds

    # ---- the refactored class reproduces the reported model and its selection criterion
    cv, sel, sel_preds = cv_and_test()
    ref = np.load(EXPERIMENTS_DIR / "variants" / "mean_reversion_cv_selected.npz")
    for h in H:
        assert np.max(np.abs(sel_preds[h] - ref[f"test_h{h}_median"])) < 1e-9
    assert abs(cv - pd.read_csv(RESULTS_DIR / "mean_reversion_cv_candidates.csv").set_index("candidate").loc["two_state/group/thb", "cv_rel"]) < 1e-12
    print(f"reproduced: cv {cv:.4f}  test MAE {sel['MAE_overall']:.2f}  RelMAE {sel['RelMAE_overall']:.4f}", flush=True)

    # ---- sensitivity of the trailing window and the deviation-state quantile
    rows = []
    for window, q in [(250, 0.5), (500, 0.5), (750, 0.5), (1000, 0.5), (750, 0.4), (750, 0.6)]:
        c, s, _ = cv_and_test(window=window, cut_q=q)
        rows.append(dict(window=window, cut_q=q, cv_RelMAE=c, **s))
        print(f"window {window:5d} q {q:.1f}: cv {c:.4f}  test MAE {s['MAE_overall']:.2f}  RelMAE {s['RelMAE_overall']:.4f}  xr {s['MAExr_overall']:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "mr_sensitivity.csv", index=False)

    # ---- the selected design fitted through 2022 only, like the TFT
    _, s22, _ = cv_and_test(label_end=val_start)
    pd.DataFrame([dict(fit_labels_through=2022, **s22)]).to_csv(RESULTS_DIR / "mr_fit_through_2022.csv", index=False)
    print(f"fitted through 2022: test MAE {s22['MAE_overall']:.2f}  RelMAE {s22['RelMAE_overall']:.4f}", flush=True)

    # ---- statistical baselines
    base = []
    one = np.array(["all"] * n)
    _, shrink, _ = cv_and_test(structure="single", cells=one)
    base.append(dict(model="global_shrinkage", **shrink))
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    for name, damped in [("ses", False), ("damped_holt", True)]:
        lvl = np.empty_like(P); trd = np.zeros_like(P); phi = np.ones(n)
        for i in range(n):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spec = dict(trend="add" if damped else None, damped_trend=damped)
                fit = ExponentialSmoothing(P[i, :test_start], initialization_method="estimated", **spec).fit()
                prm = fit.params
                kw = dict(smoothing_level=prm["smoothing_level"], optimized=False)
                init = dict(initial_level=prm["initial_level"])
                if damped:
                    kw.update(smoothing_trend=prm["smoothing_trend"], damping_trend=prm["damping_trend"])
                    init.update(initial_trend=prm["initial_trend"])
                full = ExponentialSmoothing(P[i], initialization_method="known", **spec, **init).fit(**kw)
                lvl[i] = full.level
                if damped:
                    trd[i] = full.trend; phi[i] = prm["damping_trend"]
                    # the recursion reproduces the library's own forecast at the end of training
                    cum = np.sum(phi[i] ** np.arange(1, 21))
                    assert abs(fit.forecast(20)[-1] - (fit.level[-1] + cum * fit.trend[-1])) < 1e-6 * max(1, abs(fit.level[-1]))
                else:
                    assert abs(fit.forecast(20)[-1] - fit.level[-1]) < 1e-6 * max(1, abs(fit.level[-1]))
        per_h = []
        for h in H:
            g, o = TW[h]
            cum = np.array([np.sum(phi[k] ** np.arange(1, h + 1)) for k in range(n)]) if damped else np.zeros(n)
            f = lvl[g, o] + cum[g] * trd[g, o]
            per_h.append(scores(P, g, o, h, f, rice))
        base.append(dict(model=name, **overall(per_h)))
        print(f"{name}: test MAE {base[-1]['MAE_overall']:.2f}  RelMAE {base[-1]['RelMAE_overall']:.4f}", flush=True)
    pd.DataFrame(base).to_csv(RESULTS_DIR / "mr_baselines.csv", index=False)

    # ---- effect sizes for the selected model
    rows = []
    fam_codes = np.unique(family)
    logr = {}
    for h in H:
        g, o = TW[h]; a = P[g, o + h]
        per = pd.DataFrame({"g": g, "e": np.abs(a - sel_preds[h]), "ep": np.abs(a - P[g, o])}).groupby("g").mean()
        per = per[per.ep > 0]
        d = (per.e - per.ep) / per.ep
        logr[h] = pd.Series(np.log(np.maximum(per.e, 1e-12) / per.ep), index=per.index)
        rows.append(dict(horizon=h, n=len(d), median_diff=float(d.median()), q25=float(d.quantile(0.25)), q75=float(d.quantile(0.75)),
                         share_improved=float((d < 0).mean()), RelMAE=float(np.exp(logr[h].mean()))))
    boots = []
    prods_by_fam = {f: np.flatnonzero(family == f) for f in fam_codes}
    for _ in range(2000):
        pick = np.concatenate([prods_by_fam[f] for f in rng.choice(fam_codes, len(fam_codes))])
        per_h = [np.exp(logr[h].reindex(pick).dropna().mean()) for h in H]
        boots.append([*per_h, gmean(per_h)])
    boots = np.array(boots)
    for k, h in enumerate(H):
        rows[k].update(ci_lo=float(np.percentile(boots[:, k], 2.5)), ci_hi=float(np.percentile(boots[:, k], 97.5)))
    rows.append(dict(horizon="overall", RelMAE=sel["RelMAE_overall"], ci_lo=float(np.percentile(boots[:, 4], 2.5)), ci_hi=float(np.percentile(boots[:, 4], 97.5))))
    eff = pd.DataFrame(rows)
    eff.to_csv(RESULTS_DIR / "mr_effect_sizes.csv", index=False)
    print(eff.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
