"""Every table of the mean-reversion paper.

Scores the forecaster selected by rolling-origin validation
(experiments/run_mean_reversion_cv.py) against persistence, the classical
rules, the learned reference suite, zero-shot Chronos-Bolt, both TFTs, and the
two designs chosen by single-year validation, all on the identical matched
test windows. Writes results/mr_*.csv and results/mr_parameters.json.

  mr_main.csv            per-horizon MAE, RMSE, MAPE, SMAPE, relative MAE, MAE without rice
  mr_reference_suite.csv every model in THB and scale-free terms
  mr_ablation.csv        structure x pooling, refitted through 2023 (interpretation only)
  mr_significance.csv    product-, family- and group-clustered tests with Holm correction
  mr_robustness.csv      test years, repricing thresholds, observed-only targets, without rice
  mr_intervals.csv       80% interval coverage and width
  mr_group.csv           per-group and per-family relative MAE of the forecaster and the TFT
  mr_parameters.json     fitted reversion shares of the selected forecaster
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import glob, itertools, json
import numpy as np
import pandas as pd
from scipy import stats

from paths import CHAMPION_NPZ, DATA_GLOB, EXPERIMENTS_DIR, RESULTS_DIR
from src.data.tft_loader import build_dataset
from src.models.train_tft import time_idx_for_date
from src.models.mean_reversion import MeanReversion
from experiments.run_mean_reversion import load

H = [20, 60, 120, 250]
RICE = "Wholesale Rice & Bags"
rng = np.random.default_rng(0)


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(run, 1.0)
    return adj


def smape(a, f):
    d = (np.abs(a) + np.abs(f)) / 2.0
    return float(np.mean(np.where(d == 0, 0.0, 100.0 * np.abs(f - a) / d)))


def mape(a, f):
    # cleaning casts zero prices to missing, so every actual price is positive
    assert np.all(a > 0)
    return float(np.mean(100.0 * np.abs(f - a) / a))


def per_product(a, f, p, g):
    return pd.DataFrame({"g": g, "e": np.abs(a - f), "ep": np.abs(a - p)}).groupby("g").mean()


def rel_mae(a, f, p, g):
    per = per_product(a, f, p, g); per = per[per.ep > 0]
    return float(np.exp(np.mean(np.log(np.maximum(per.e, 1e-12) / per.ep))))


def gmean(x):
    return float(np.exp(np.mean(np.log(x))))


def main():
    P, val_start, test_start, codes, group, family, level = load()
    full_df, _ = build_dataset()
    full_df["date"] = pd.to_datetime(full_df["date"])
    y2025 = time_idx_for_date(full_df, "2025-01-01")
    pos = {p: i for i, p in enumerate(codes)}
    T = P.shape[1]
    rice = group == RICE
    summ = pd.read_csv(RESULTS_DIR / "eda_product_summary.csv").set_index("product_id")
    move = summ["move_rate"].reindex(codes).to_numpy()
    pools = {"global": np.array(["all"] * len(codes), dtype=object), "group": group, "family": family}

    sel = json.load(open(RESULTS_DIR / "mean_reversion_cv_selected.json"))["selected"]
    structure, pool, objective = sel.split("/")
    mr = np.load(EXPERIMENTS_DIR / "variants" / "mean_reversion_cv_selected.npz")
    tft = np.load(CHAMPION_NPZ)

    # refit the selected design through 2023 and confirm it reproduces the saved forecasts
    model = MeanReversion(P, level, pools[pool], structure, objective)
    W = {}
    for h in H:
        model.fit(h, test_start)
        g = mr[f"test_h{h}_group"].astype(int); li = mr[f"test_h{h}_label_idx"].astype(int); o = li - h
        assert np.array_equal(g, tft[f"test_h{h}_group"].astype(int))
        f = model.predict(h, g, o)
        assert np.max(np.abs(f - mr[f"test_h{h}_median"])) < 1e-9
        W[h] = dict(g=g, li=li, o=o, a=P[g, li], persistence=P[g, o], mean_reversion=f,
                    hw_tft=tft[f"test_h{h}_median"], seasonal_naive=P[g, li - 250],
                    drift=P[g, o] + h * (P[g, o] - P[g, 0]) / np.maximum(o, 1))
    json.dump({"selected": sel, "params": {str(h): {f"{c}|{s}": v for (c, s), v in model.params[h].items()} for h in H}},
              open(RESULTS_DIR / "mr_parameters.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    # designs chosen by the single-year validation protocol, same windows
    for name, fn in [("mr_single_year_base", "mean_reversion_selected.npz"), ("mr_single_year_extended", "mean_reversion_v2_selected.npz")]:
        z = np.load(EXPERIMENTS_DIR / "variants" / fn)
        for h in H:
            assert np.array_equal(z[f"test_h{h}_group"].astype(int), W[h]["g"])
            W[h][name] = z[f"test_h{h}_median"]

    uz = np.load(EXPERIMENTS_DIR / "tft_hw_quantile_gamma_0_0_predictions.npz")
    for h in H:
        assert np.array_equal(uz[f"test_h{h}_group"].astype(int), W[h]["g"])
        W[h]["tft_unweighted"] = uz[f"test_h{h}_median"]
    assert abs(np.mean([np.mean(np.abs(W[h]["a"] - W[h]["tft_unweighted"])) for h in H]) - 72.85) < 0.01

    c = np.load(EXPERIMENTS_DIR / "chronos_zeroshot_predictions.npz", allow_pickle=True)
    key = pd.Series(np.arange(len(c["origin"])),
                    index=pd.MultiIndex.from_arrays([[pos[p] for p in c["product_id"]], c["origin"].astype(int)]))
    for h in H:
        idx = key.reindex(pd.MultiIndex.from_arrays([W[h]["g"], W[h]["o"]])).to_numpy()
        assert not np.isnan(idx.astype(float)).any()
        idx = idx.astype(int)
        assert np.allclose(c[f"actual_h{h}"][idx], W[h]["a"], atol=1e-2)
        W[h]["chronos_bolt"] = c[f"median_h{h}"][idx]

    # zero-shot TimesFM 2.5 and 3.0 (experiments/run_timesfm_zeroshot.py), same windows
    timesfm_models = []
    # plus Chronos-2 zero-shot and LoRA fine-tuned (run_chronos2.py) and TimesFM 2.5 LoRA (run_timesfm_finetune.py)
    for name, fn in [("timesfm_2p5", "timesfm_2p5_zeroshot_predictions.npz"), ("timesfm_3p0", "timesfm_3p0_zeroshot_predictions.npz"),
                     ("timesfm_2p5_lora", "timesfm_lora_predictions.npz"), ("chronos2", "chronos2_zeroshot_predictions.npz"),
                     ("chronos2_lora", "chronos2_finetuned_predictions.npz")]:
        fz = EXPERIMENTS_DIR / fn
        if fz.exists():
            tz = np.load(fz)
            for h in H:
                assert np.array_equal(tz[f"test_h{h}_group"].astype(int), W[h]["g"])
                assert np.allclose(tz[f"test_h{h}_actual"], W[h]["a"], atol=1e-3)
                W[h][name] = tz[f"test_h{h}_median"]
            timesfm_models.append(name)

    pub = pd.read_csv(RESULTS_DIR / "extra_baselines.csv").set_index("model")
    for m_, k in [("seasonal_naive", "snaive"), ("drift", "drift")]:
        got = np.mean([np.mean(np.abs(W[h]["a"] - W[h][m_])) for h in H])
        assert abs(got - pub.loc[k, "mae_overall"]) < 0.01

    models = ["persistence", "mean_reversion", "mr_single_year_base", "mr_single_year_extended", "hw_tft",
              "tft_unweighted", "seasonal_naive", "drift", "chronos_bolt"] + timesfm_models

    # ---- main table and window-level suite
    main_rows, suite = [], []
    for m_ in models:
        rec = {"model": m_}
        for h in H:
            w = W[h]; a, f, p = w["a"], w[m_], w["persistence"]; xr = ~rice[w["g"]]
            rec.update({f"MAE_{h}": float(np.mean(np.abs(a - f))), f"RMSE_{h}": float(np.sqrt(np.mean((a - f) ** 2))),
                        f"MAPE_{h}": mape(a, f), f"SMAPE_{h}": smape(a, f), f"RelMAE_{h}": rel_mae(a, f, p, w["g"]),
                        f"MAExr_{h}": float(np.mean(np.abs(a[xr] - f[xr]))),
                        f"RelMAExr_{h}": rel_mae(a[xr], f[xr], p[xr], w["g"][xr])})
        for k in ["MAE", "RMSE", "MAPE", "SMAPE", "MAExr"]:
            rec[f"{k}_overall"] = float(np.mean([rec[f"{k}_{h}"] for h in H]))
        for k in ["RelMAE", "RelMAExr"]:
            rec[f"{k}_overall"] = gmean([rec[f"{k}_{h}"] for h in H])
        main_rows.append(rec)
        suite.append({"model": m_, "source": "window", **{k: rec[k] for k in
                      ["MAE_overall", "MAExr_overall", "RelMAE_overall", "RelMAExr_overall", "RMSE_overall", "MAPE_overall", "SMAPE_overall"]},
                      **{f"RelMAE_{h}": rec[f"RelMAE_{h}"] for h in H}})
    pd.DataFrame(main_rows).to_csv(RESULTS_DIR / "mr_main.csv", index=False)

    m = pd.read_csv(RESULTS_DIR / "metrics_by_horizon.csv"); m = m[m.horizon.isin(H)]
    base = m[m.model_name == "baseline"].set_index(["product_id", "horizon"])["MAE"]
    rice_ids = set(np.asarray(codes)[rice])
    for (par, mod), d in m[(m.model_name != "baseline")].groupby(["paradigm", "model_name"]):
        d = d.set_index(["product_id", "horizon"])["MAE"]
        rec = {"model": f"{par}_{mod}", "source": "per_product"}
        rr, rx, mae, maex = [], [], [], []
        for h in H:
            dh = d.xs(h, level="horizon"); bh = base.xs(h, level="horizon").reindex(dh.index)
            ok = bh > 0; xr = ~dh.index.isin(rice_ids)
            r = gmean(np.maximum(dh[ok], 1e-12) / bh[ok]); rec[f"RelMAE_{h}"] = r; rr.append(r)
            rx.append(gmean(np.maximum(dh[ok & xr], 1e-12) / bh[ok & xr])); mae.append(dh.mean()); maex.append(dh[xr].mean())
        rec.update(MAE_overall=float(np.mean(mae)), MAExr_overall=float(np.mean(maex)),
                   RelMAE_overall=gmean(rr), RelMAExr_overall=gmean(rx))
        suite.append(rec)
    pd.DataFrame(suite).sort_values("MAE_overall").to_csv(RESULTS_DIR / "mr_reference_suite.csv", index=False)

    # ---- ablation: structure x pooling with the selected objective, refitted through 2023
    abl = []
    for st, pl in itertools.product(["single", "two", "state", "two_state"], ["global", "group", "family"]):
        mm = MeanReversion(P, level, pools[pl], st, objective)
        rec = {"structure": st, "pooling": pl}
        rr = []
        for h in H:
            mm.fit(h, test_start); w = W[h]
            f = mm.predict(h, w["g"], w["o"]); xr = ~rice[w["g"]]
            r = rel_mae(w["a"], f, w["persistence"], w["g"]); rr.append(r)
            rec.update({f"RelMAE_{h}": r, f"MAE_{h}": float(np.mean(np.abs(w["a"] - f))),
                        f"MAExr_{h}": float(np.mean(np.abs(w["a"][xr] - f[xr])))})
        rec["RelMAE_overall"] = gmean(rr)
        rec["MAE_overall"] = float(np.mean([rec[f"MAE_{h}"] for h in H]))
        rec["MAExr_overall"] = float(np.mean([rec[f"MAExr_{h}"] for h in H]))
        abl.append(rec)
        print(f"ablation {st:10s} {pl:7s} RelMAE {rec['RelMAE_overall']:.4f}  MAE {rec['MAE_overall']:.2f}  MAExr {rec['MAExr_overall']:.2f}", flush=True)
    pd.DataFrame(abl).to_csv(RESULTS_DIR / "mr_ablation.csv", index=False)

    # ---- significance of the relative per-product differential, three clusterings, Holm across horizons
    sig = []
    for h in H:
        w = W[h]
        per = per_product(w["a"], w["mean_reversion"], w["persistence"], w["g"]); per = per[per.ep > 0]
        d = (per.e - per.ep) / per.ep
        for lvl, lab in [("product", per.index.to_numpy()), ("family", family[per.index]), ("group", group[per.index])]:
            cl = d.groupby(lab).mean()
            byc = {k: d[lab == k].to_numpy() for k in cl.index}
            boots = [np.concatenate([byc[k] for k in rng.choice(cl.index, len(cl))]).mean() for _ in range(2000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            sig.append(dict(horizon=h, level=lvl, n=len(cl), mean_rel_diff=float(cl.mean()), wins=int((cl < 0).sum()),
                            p_t=float(stats.ttest_1samp(cl, 0).pvalue), p_wilcoxon=float(stats.wilcoxon(cl).pvalue),
                            ci_lo=float(lo), ci_hi=float(hi)))
    sig = pd.DataFrame(sig)
    for lvl in sig.level.unique():
        k = sig.level == lvl
        sig.loc[k, "p_t_holm"] = holm(sig.loc[k, "p_t"]); sig.loc[k, "p_wilcoxon_holm"] = holm(sig.loc[k, "p_wilcoxon"])
    sig.to_csv(RESULTS_DIR / "mr_significance.csv", index=False)

    # ---- robustness
    dates = full_df.drop_duplicates("time_idx").set_index("time_idx")["date"].sort_index()
    d2i = {d.strftime("%Y-%m-%d"): i for i, d in dates.items()}
    OBS = np.zeros((len(codes), T), bool)
    for fp in glob.glob(DATA_GLOB):
        js = json.load(open(fp, encoding="utf-8-sig")); pid = js.get("product_id")
        if pid in pos:
            for r in js.get("price_list", []):
                v, k = r.get("price_min"), str(r.get("date", ""))[:10]
                if v not in (None, 0, 0.0) and k in d2i:
                    OBS[pos[pid], d2i[k]] = True
    subsets = {"all": lambda w: np.ones(len(w["a"]), bool), "without_rice": lambda w: ~rice[w["g"]],
               "labels_2024": lambda w: w["li"] < y2025, "labels_2025": lambda w: w["li"] >= y2025,
               "repricing_2pct": lambda w: move[w["g"]] >= 0.02, "repricing_5pct": lambda w: move[w["g"]] >= 0.05,
               "repricing_10pct": lambda w: move[w["g"]] >= 0.10, "observed_targets": lambda w: OBS[w["g"], w["li"]]}
    rob = []
    for sname, fn in subsets.items():
        rec = {"subset": sname, "n_products": int(len(np.unique(W[250]["g"][fn(W[250])])))}
        for mname in ["persistence", "mean_reversion", "hw_tft"]:
            maes, rels = [], []
            for h in H:
                w = W[h]; k = fn(w)
                maes.append(float(np.mean(np.abs(w["a"][k] - w[mname][k]))))
                rels.append(rel_mae(w["a"][k], w[mname][k], w["persistence"][k], w["g"][k]))
                rec[f"{mname}_MAE_{h}"] = maes[-1]; rec[f"{mname}_RelMAE_{h}"] = rels[-1]
            rec[f"{mname}_MAE_overall"] = float(np.mean(maes)); rec[f"{mname}_RelMAE_overall"] = gmean(rels)
        rob.append(rec)
    pd.DataFrame(rob).to_csv(RESULTS_DIR / "mr_robustness.csv", index=False)

    # ---- 80% intervals from out-of-sample 2023 relative residuals, per pooling cell and horizon
    cells = pools[pool]
    iv = []
    for h in H:
        vg = mr[f"val_h{h}_group"].astype(int)
        res = (mr[f"val_h{h}_actual"] - mr[f"val_h{h}_median"]) / np.maximum(mr[f"val_h{h}_median"], 1e-9)
        q = pd.DataFrame({"c": cells[vg], "r": res}).groupby("c")["r"].quantile([0.1, 0.9]).unstack()
        w = W[h]; f = w["mean_reversion"]; cc = cells[w["g"]]
        lo = f * (1 + q.loc[cc, 0.1].to_numpy()); hi = f * (1 + q.loc[cc, 0.9].to_numpy())
        cov = (w["a"] >= lo) & (w["a"] <= hi)
        byc = pd.Series(cov).groupby(cc).mean()
        iv.append(dict(horizon=h, coverage=float(cov.mean()), coverage_without_rice=float(cov[~rice[w["g"]]].mean()),
                       mean_width_thb=float(np.mean(hi - lo)), median_relative_width=float(np.median((hi - lo) / f)),
                       cell_coverage_min=float(byc.min()), cell_coverage_max=float(byc.max())))
    pd.DataFrame(iv).to_csv(RESULTS_DIR / "mr_intervals.csv", index=False)

    # ---- per-group and per-family view
    rows = []
    for level_name, lab in [("group", group), ("family", family)]:
        for c_ in sorted(np.unique(lab)):
            rec = {"level": level_name, "cell": c_, "n_products": int((lab == c_).sum())}
            for h in H:
                w = W[h]; k = lab[w["g"]] == c_
                rec[f"RelMAE_{h}"] = rel_mae(w["a"][k], w["mean_reversion"][k], w["persistence"][k], w["g"][k])
                rec[f"tft_RelMAE_{h}"] = rel_mae(w["a"][k], w["hw_tft"][k], w["persistence"][k], w["g"][k])
            rows.append(rec)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "mr_group.csv", index=False)

    pd.set_option("display.width", 240)
    mt = pd.DataFrame(main_rows).set_index("model")
    print(mt[[f"MAE_{h}" for h in H] + ["MAE_overall", "RMSE_overall", "SMAPE_overall", "RelMAE_overall", "MAExr_overall", "RelMAExr_overall"]].round(3).to_string())
    print(pd.DataFrame(suite).sort_values("MAE_overall")[["model", "MAE_overall", "RelMAE_overall", "MAExr_overall", "RelMAExr_overall"]].round(3).to_string(index=False))
    print(sig.round(4).to_string(index=False))
    print(pd.DataFrame(iv).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
