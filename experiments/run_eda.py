"""Full EDA redo on the raw crop JSONs.

Mirrors CropDataLoader's cleaning (zero->NaN, rolling IQR, ffill/bfill, weekday
skeleton) via the loader's own public methods, but bypasses the volatility gate
so all active series are profiled. Outputs:
  - eda_stats.json (all headline numbers)
  - results/eda_product_summary.csv (per-product profile)
  - four paper figures (same filenames) into paper/images/
"""
import os, sys, io, glob, json, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# make the repository root importable regardless of how this is invoked
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from src.data.loader import CropDataLoader, EmptySeriesError

from paths import ROOT as _ROOT

PKG = str(_ROOT)
from paths import IMAGE_DIR as _IMAGE_DIR
IMG = str(_IMAGE_DIR)
from paths import RESULTS_DIR as _RESULTS_DIR

SCRATCH = str(_RESULTS_DIR)

GROUP_EN = {
    "เนื้อสัตว์": "Meat & Poultry", "สัตว์น้ำ": "Aquatic & Marine",
    "ผักสด": "Fresh Vegetables", "ผัก-ผลไม้อินทรีย์": "Organic Veg & Fruits",
    "ผลไม้": "Fruits", "พืชอาหาร": "Food Crops",
    "พืชน้ำมันและน้ำมันพืช": "Oil Crops & Oils",
    "ราคาขายส่งข้าว ผลิตภัณฑ์ข้าวและกระสอบป่าน": "Wholesale Rice & Bags",
    "ราคาขายส่งข้าวสารให้ร้านขายปลีก": "Wholesale Rice to Retailers",
    "ราคาขายปลีกข้าวสาร": "Retail Rice", "พืชไร่": "Field Crops",
    "อาหารสัตว์และวัตถุดิบอาหารสัตว์": "Feed & Raw Materials",
    "": "Miscellaneous / Unknown", None: "Miscellaneous / Unknown",
}
CAT_EN = {"ขายปลีก": "Retail", "ขายส่ง": "Wholesale"}
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]

loader = CropDataLoader()
files = sorted(glob.glob(__import__("paths").DATA_GLOB))
print(f"files: {len(files)}", flush=True)

rows, series_store = [], {}
total_obs = zero_obs = 0
for i, fp in enumerate(files):
    if i % 100 == 0:
        print(f"  {i}/{len(files)}", flush=True)
    # raw zero-share accounting (all files, before any cleaning)
    with open(fp, "r", encoding="utf-8-sig") as f:
        raw = json.load(f)
    meta_group = raw.get("group_name") or ""
    meta_cat = raw.get("category_name") or ""
    pid = raw.get("product_id", fp.split("\\")[-1][:-5])
    vals = []
    for rec in raw.get("price_list", []):
        for k in ("price_min", "price_max"):
            v = rec.get(k)
            if v is not None:
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    pass
    total_obs += len(vals)
    zero_obs += sum(1 for v in vals if v == 0)

    row = {"product_id": pid, "group": GROUP_EN.get(meta_group, meta_group),
           "category": CAT_EN.get(meta_cat, meta_cat or "Unknown")}
    try:
        df_raw, meta = loader.load_json(fp)
        df = loader.align_to_skeleton(loader.clean_outliers(df_raw))
        row["status"] = "active"
        pmax = df["price_max"].to_numpy(float)
        pmin = df["price_min"].to_numpy(float)
        d = np.diff(pmax[~np.isnan(pmax)])
        row["move_rate"] = float(np.sum(d != 0) / len(d)) if len(d) else 0.0
        row["mean_price"] = float(np.nanmean(pmin))
        row["raw_reports"] = len(df_raw)
        try:
            row["adf_p_min"] = float(adfuller(pmin[~np.isnan(pmin)], autolag="AIC")[1])
        except Exception:
            row["adf_p_min"] = np.nan
        try:
            row["adf_p_max"] = float(adfuller(pmax[~np.isnan(pmax)], autolag="AIC")[1])
        except Exception:
            row["adf_p_max"] = np.nan
        series_store[pid] = pd.Series(pmin, index=pd.to_datetime(df["date"]))
    except EmptySeriesError:
        row["status"] = "empty"
    rows.append(row)

prof = pd.DataFrame(rows)
prof.to_csv(PKG + r"\results\eda_product_summary.csv", index=False)
act = prof[prof.status == "active"].copy()
act["stat_min"] = act.adf_p_min < 0.05
act["stat_max"] = act.adf_p_max < 0.05

stats = {
    "n_files": len(files),
    "n_empty": int((prof.status == "empty").sum()),
    "n_active": int(len(act)),
    "zero_share_pct": round(100 * zero_obs / total_obs, 2),
    "n_pass_1pct": int((act.move_rate >= 0.01).sum()),
    "n_pass_5pct": int((act.move_rate >= 0.05).sum()),
    "move_rate_median": round(float(act.move_rate.median()), 4),
    "stationary_min_pct_all": round(100 * act.stat_min.mean(), 1),
    "stationary_max_pct_all": round(100 * act.stat_max.mean(), 1),
}
# Table 1: by group
t1 = prof.groupby("group").status.value_counts().unstack(fill_value=0)
t1["total"] = t1.sum(axis=1)
stats["table1"] = t1.to_dict()
# Table 2: stationarity by category
t2 = act.groupby("category").agg(n=("product_id", "count"),
                                 stat_min=("stat_min", "mean"), stat_max=("stat_max", "mean"))
stats["table2"] = {k: {"n": int(v["n"]), "min_pct": round(100 * v["stat_min"], 1),
                       "max_pct": round(100 * v["stat_max"], 1)} for k, v in t2.iterrows()}
json.dump(stats, open(SCRATCH + r"\eda_stats.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in stats.items() if not isinstance(v, dict)}, indent=1), flush=True)
print(t2.to_string(), flush=True)

plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})

# ---- Figure 1: mean ACF/PACF across all active series, by product group ----
from statsmodels.tsa.stattools import acf as _acf, pacf as _pacf
LAGS = 40
g_acf, g_pacf = {}, {}
for pid, s in series_store.items():
    grp = act.loc[act.product_id == pid, "group"]
    if grp.empty:
        continue
    ss = s.dropna()
    if len(ss) < LAGS + 5 or ss.nunique() < 3:
        continue
    try:
        a = _acf(ss, nlags=LAGS, fft=True); p = _pacf(ss, nlags=LAGS, method="ywm")
    except Exception:
        continue
    g_acf.setdefault(grp.iloc[0], []).append(a)
    g_pacf.setdefault(grp.iloc[0], []).append(p)
# small multiples: one panel per group so every group is identifiable (12 hues would not be
# colorblind-safe, and the curves cross, so a single overlaid panel cannot be direct-labelled)
order = sorted(g_acf, key=lambda g: -np.mean([v[LAGS] for v in g_acf[g]]))
A = np.vstack([np.mean(g_acf[g], axis=0) for g in order])
P = np.vstack([np.mean(g_pacf[g], axis=0) for g in order])
Nn = [len(g_acf[g]) for g in order]
ref = A.mean(axis=0)
lags = np.arange(LAGS + 1)
C_ACF, C_PACF, C_REF = OKABE[0], OKABE[1], "#BBBBBB"
fig, axes = plt.subplots(3, 4, figsize=(12, 7.2), sharex=True, sharey=True)
for i, (g, ax) in enumerate(zip(order, axes.ravel())):
    ax.plot(lags, ref, color=C_REF, linewidth=1.2, zorder=1)
    ax.plot(lags, A[i], color=C_ACF, linewidth=2.0, zorder=3)
    ax.plot(lags, P[i], color=C_PACF, linewidth=1.5, zorder=2)
    ax.axhline(0, color="#444", linewidth=0.6, zorder=0)
    ax.set_title(f"{g}  (n={Nn[i]})", fontsize=9, pad=4)
    # label sits in the empty band between the PACF (~0) and the lowest ACF curve
    ax.annotate(f"ACF$_{{40}}$ = {A[i, LAGS]:.2f}", (0.96, 0.30), xycoords="axes fraction",
                ha="right", va="center", fontsize=8, color=C_ACF)
    ax.set_xlim(0, LAGS); ax.set_ylim(-0.2, 1.05); ax.set_yticks([0, 0.5, 1.0])
for ax in axes[-1]:
    ax.set_xlabel("Lag (business days)")
for ax in axes[:, 0]:
    ax.set_ylabel("Correlation")
handles = [plt.Line2D([], [], color=C_ACF, lw=2.0, label="ACF (this group)"),
           plt.Line2D([], [], color=C_PACF, lw=1.5, label="PACF (this group)"),
           plt.Line2D([], [], color=C_REF, lw=1.2, label="Mean ACF, all groups")]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.005))
fig.suptitle("Autocorrelation structure of daily crop prices by product group (levels), ordered by persistence",
             fontsize=11.5)
fig.tight_layout(rect=[0, 0.035, 1, 0.97])
fig.savefig(os.path.join(IMG, "acf_pacf.png"), dpi=150)
plt.close(fig)
picks = [type("R", (), {"product_id": order[0]}), type("R", (), {"product_id": order[-1]})]

# ---- Figure 2: indexed price trends of six largest groups ----
top_groups = act.group.value_counts().head(6).index.tolist()
fig, ax = plt.subplots(figsize=(10, 5))
for g, c in zip(top_groups, OKABE):
    pids = act[act.group == g].product_id
    idxed = []
    for pid in pids:
        s = series_store[pid]
        base = s["2018"].mean()
        if base and not np.isnan(base) and base > 0:
            idxed.append(s / base * 100)
    gi = pd.concat(idxed, axis=1).mean(axis=1).resample("ME").mean()
    ax.plot(gi.index, gi.values, color=c, linewidth=2, label=f"{g} (n={len(pids)})")
    ax.annotate(g, (gi.index[-1], gi.values[-1]), xytext=(4, 0), textcoords="offset points",
                color=c, fontsize=8, va="center")
ax.set_ylabel("Mean indexed price (2018 = 100)")
ax.set_title("Indexed price trajectories of the six largest product groups (monthly means)")
ax.legend(fontsize=8, ncol=2, loc="upper left")
ax.margins(x=0.12)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "price_trend.png"), dpi=150)
plt.close(fig)

# ---- Figure 3: stationarity by group ----
g3 = act.groupby("group").agg(n=("product_id", "count"), smin=("stat_min", "mean"),
                              smax=("stat_max", "mean")).sort_values("smin")
fig, ax = plt.subplots(figsize=(10, 6.5))
y = np.arange(len(g3))
ax.barh(y + 0.2, 100 * g3.smin, height=0.38, color=OKABE[0], label="Minimum price")
ax.barh(y - 0.2, 100 * g3.smax, height=0.38, color=OKABE[1], label="Maximum price")
ax.set_yticks(y, [f"{g} (n={int(r.n)})" for g, r in g3.iterrows()], fontsize=9)
ax.set_xlabel("Share of series stationary in levels (ADF, p < 0.05), %")
ax.set_title("ADF stationarity rates of active price series by product group")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(IMG, "stationarity_by_group.png"), dpi=150)
plt.close(fig)

# ---- Figure 4: correlation heatmap of the 40 most liquid products, by commodity family ----
# Product names are Thai; EN maps product_id -> (concise English label, commodity family).
# The families are annotated from the product names because the dataset's own group_name
# lumps rubber, cassava and grain together as "Field Crops", which hides the real blocks.
EN = {
 "W16005": ("Cassava chips, FOB export", "Cassava"),
 "W16007": ("Tapioca starch premium, FOB export", "Cassava"),
 "W16044": ("Cassava alcohol, China price", "Cassava"),
 "W16017": ("Feed maize, feed mill", "Grain"),
 "W16037": ("Feed maize, CBOT futures", "Grain"),
 "W16038": ("Wheat, CBOT futures", "Grain"),
 "W16023": ("Rubber sheet USS3, farm gate Surat Thani", "Rubber"),
 "W16024": ("Rubber sheet USS3, farm gate Songkhla", "Rubber"),
 "W16025": ("Rubber sheet USS3, farm gate Nakhon Si Th.", "Rubber"),
 "W16026": ("Rubber sheet USS3, central mkt Surat Thani", "Rubber"),
 "W16027": ("Rubber sheet USS3, central mkt Songkhla", "Rubber"),
 "W16028": ("Rubber sheet USS3, central mkt Nakhon Si Th.", "Rubber"),
 "W16029": ("Rubber scrap 100%, Songkhla", "Rubber"),
 "W16030": ("Rubber sheet RSS3, FOB Laem Chabang", "Rubber"),
 "W16033": ("Rubber sheet RSS3, central mkt Songkhla", "Rubber"),
 "W16034": ("Rubber cup lump 100%, opening price", "Rubber"),
 "W16036": ("Fresh latex, opening price", "Rubber"),
 "P13022": ("Yardlong bean, mixed grade", "Vegetable"),
 "W13018": ("Coriander", "Vegetable"),
 "W13022": ("Chinese celery", "Vegetable"),
 "W18020": ("Crude palm oil, grade A (mesocarp)", "Palm oil"),
 "W18021": ("Crude palm oil, grade B (mixed)", "Palm oil"),
 "W18024": ("RBD palm olein, domestic", "Palm oil"),
 "W18025": ("RBD palm stearin, domestic", "Palm oil"),
 "W18036": ("Crude palm oil, Malaysia (a)", "Palm oil"),
 "W18037": ("Crude palm oil, Malaysia (b)", "Palm oil"),
 "W18038": ("RBD palm oil, Malaysia (a)", "Palm oil"),
 "W18039": ("RBD palm oil, Malaysia (b)", "Palm oil"),
 "W18040": ("RBD palm olein, Malaysia (a)", "Palm oil"),
 "W18041": ("RBD palm olein, Malaysia (b)", "Palm oil"),
 "W18042": ("RBD palm stearin, Malaysia (a)", "Palm oil"),
 "W18043": ("RBD palm stearin, Malaysia (b)", "Palm oil"),
 "W18086": ("Palm fruit bunch 18%, national", "Palm fruit"),
 "W18087": ("Palm fruit bunch 18%, Krabi", "Palm fruit"),
 "W18088": ("Palm fruit bunch 18%, Surat Thani", "Palm fruit"),
 "W18089": ("Palm fruit bunch 18%, Chumphon", "Palm fruit"),
 "W18090": ("Palm fruit bunch 18%, Trang", "Palm fruit"),
 "W18091": ("Palm fruit bunch 18%, Satun", "Palm fruit"),
 "W18095": ("Palm fruit bunch 18%, Nakhon Si Th.", "Palm fruit"),
 "W18096": ("Palm fruit bunch 18%, Phang Nga", "Palm fruit"),
}
FAM_ORDER = ["Cassava", "Grain", "Rubber", "Vegetable", "Palm oil", "Palm fruit"]
top40 = list(liquid.head(40).product_id)
if set(top40) == set(EN):
    ordered = sorted(top40, key=lambda p: (FAM_ORDER.index(EN[p][1]), p))
    lab = [EN[p][0] for p in ordered]; fams = [EN[p][1] for p in ordered]
else:  # liquidity ranking shifted; fall back to IDs so the run still completes
    print("WARN: top-40 set changed; falling back to product_id labels", flush=True)
    ordered = sorted(top40, key=lambda p: (act.set_index('product_id').group[p], p))
    lab = ordered; fams = [act.set_index('product_id').group[p] for p in ordered]
mat = pd.concat([series_store[p] for p in ordered], axis=1)
mat.columns = ordered
corr = mat.corr()
V = corr.values
nn = len(ordered)
fig, ax = plt.subplots(figsize=(13.5, 11.5))
im = ax.imshow(V, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(nn), lab, rotation=90, fontsize=6.5)
ax.set_yticks(range(nn), lab, fontsize=6.5)
bounds = [i + 0.5 for i in range(nn - 1) if fams[i] != fams[i + 1]]
for b in bounds:
    ax.axhline(b, color="black", linewidth=1.0); ax.axvline(b, color="black", linewidth=1.0)
edges = [-0.5] + bounds + [nn - 0.5]
for k in range(len(edges) - 1):
    ax.text(nn + 0.4, (edges[k] + edges[k + 1]) / 2, sorted(set(fams), key=fams.index)[k],
            rotation=270, va="center", ha="left", fontsize=8.5, fontweight="bold")
fig.colorbar(im, ax=ax, shrink=0.62, pad=0.12, label="Pearson correlation (price levels)")
ax.set_title("Cross-product price correlations — 40 most liquid series, grouped by commodity family", pad=12)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "correlation_heatmap.png"), dpi=150)
plt.close(fig)

# ---- co-movement statistics ----
iu = np.triu_indices(nn, 1)
r40 = V[iu]
same_fam = np.array([[fams[i] == fams[j] for j in range(nn)] for i in range(nn)])[iu]
ret40 = mat.pct_change().replace([np.inf, -np.inf], np.nan).corr().values[iu]
stats2 = {
    "top40_levels_mean": round(float(np.mean(r40)), 3),
    "top40_levels_share_above_05": round(float(np.mean(r40 > 0.5)), 3),
    "top40_levels_within_family": round(float(np.mean(r40[same_fam])), 3),
    "top40_levels_cross_family": round(float(np.mean(r40[~same_fam])), 3),
    "top40_returns_mean": round(float(np.nanmean(ret40)), 3),
    "top40_returns_within_family": round(float(np.nanmean(ret40[same_fam])), 3),
    "top40_returns_cross_family": round(float(np.nanmean(ret40[~same_fam])), 3),
}
# dataset-wide over the 404-crop modelling universe (the numbers the paper quotes)
uni = [p for p in act.product_id if act.set_index("product_id").move_rate[p] >= 0.01]
matu = pd.concat([series_store[p] for p in uni], axis=1); matu.columns = uni
gu = np.array([act.set_index("product_id").group[p] for p in uni])
levu = matu.corr().values
retu = matu.pct_change().replace([np.inf, -np.inf], np.nan).corr().values
iuu = np.triu_indices(len(uni), 1)
sameg = (gu[:, None] == gu[None, :])[iuu]
for nm, M in [("levels", levu), ("returns", retu)]:
    rr = M[iuu]; ok = ~np.isnan(rr)
    stats2[f"universe_{nm}_mean"] = round(float(np.nanmean(rr)), 3)
    stats2[f"universe_{nm}_median"] = round(float(np.nanmedian(rr)), 3)
    stats2[f"universe_{nm}_share_above_05"] = round(float(np.mean(rr[ok] > 0.5)), 3)
    stats2[f"universe_{nm}_within_group"] = round(float(np.nanmean(rr[sameg & ok])), 3)
    stats2[f"universe_{nm}_cross_group"] = round(float(np.nanmean(rr[~sameg & ok])), 3)
stats2["universe_n_products"] = len(uni)
stats2["universe_n_pairs"] = int(len(iuu[0]))
print(json.dumps(stats2, indent=1), flush=True)
json.dump(stats2, open(SCRATCH + r"\eda_stats2.json", "w"), indent=1)
print("figures saved", flush=True)
