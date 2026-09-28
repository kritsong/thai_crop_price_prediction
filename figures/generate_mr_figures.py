"""Figures for the mean-reversion results, from results/mr_*.csv and the matched test windows.

  error_concentration.png   share of products vs share of persistence error, by family
  reversion_map.png         fitted share of the gap closed, by group, horizon and gap size
  relmae_by_horizon.png     relative MAE against persistence by horizon
  group_relmae_h250.png     per-group relative MAE at one year
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import CHAMPION_NPZ, IMAGE_DIR, RESULTS_DIR

# validated reference palette, light mode (print)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
H = [20, 60, 120, 250]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def short(f):
    g, c = f.split(" | ")
    return f"{g} ({'W' if c == 'Wholesale' else 'R'})"


def error_concentration():
    z = np.load(CHAMPION_NPZ)
    summ = pd.read_csv(RESULTS_DIR / "eda_product_summary.csv").set_index("product_id")
    fam_of = pd.read_csv(RESULTS_DIR / "mr_group.csv").query("level == 'family'")["cell"]
    codes = np.sort(summ.index[summ.index.isin(pd.read_csv(RESULTS_DIR / "metrics_by_horizon.csv").product_id.unique())])
    family = (summ["group"].astype(str) + " | " + summ["category"].astype(str)).reindex(codes).to_numpy()
    err = []
    for h in H:
        g = z[f"test_h{h}_group"].astype(int)
        err.append(pd.Series(np.abs(z[f"test_h{h}_actual"] - z[f"test_h{h}_persistence"])).groupby(family[g]).sum())
    err = sum(err)
    d = pd.DataFrame({"products": pd.Series(family).value_counts(normalize=True),
                      "error": err / err.sum()}).reindex(sorted(fam_of)).fillna(0).sort_values("error")
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    y = np.arange(len(d)); bh = 0.38
    ax.barh(y + bh / 2 + 0.01, 100 * d.products, height=bh, color=BLUE, label="Share of products")
    ax.barh(y - bh / 2 - 0.01, 100 * d.error, height=bh, color=ORANGE, label="Share of persistence error (THB)")
    ax.set_yticks(y, [short(f) for f in d.index])
    ax.set_xlabel("Percent of total")
    ax.grid(axis="y", visible=False)
    top = d.index[-1]
    ax.text(100 * d.error[top], len(d) - 1 + 0.62, f"{100 * d.products[top]:.0f}% of products, {100 * d.error[top]:.0f}% of error",
            ha="right", va="bottom", color=INK, fontsize=8.5)
    ax.set_ylim(-0.6, len(d) - 1 + 1.1)
    ax.legend(loc="lower right")
    fig.savefig(IMAGE_DIR / "error_concentration.png"); plt.close(fig)
    return d


def reversion_map():
    import json
    j = json.load(open(RESULTS_DIR / "mr_parameters.json", encoding="utf-8"))["params"]
    groups = sorted({k.split("|")[0] for k in j["20"]})
    tot = {s: np.array([[sum(j[str(h)][f"{g}|{s}"]) for h in H] for g in groups]) for s in ["small", "large"]}
    lng = {s: np.array([[j[str(h)][f"{g}|{s}"][1] for h in H] for g in groups]) for s in ["small", "large"]}
    order = np.argsort(-(tot["small"].sum(1) + tot["large"].sum(1)))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#f4f8fd"] + SEQ)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, s, title in zip(axes, ["small", "large"], ["Gap at or below the group median", "Gap above the group median"]):
        v = tot[s][order]; b = lng[s][order]
        im = ax.imshow(v, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(v.shape[0]):
            for k in range(v.shape[1]):
                lab = f"{v[i, k]:.2f}" + (" L" if b[i, k] > v[i, k] / 2 and v[i, k] > 0 else "")
                ax.text(k, i, lab, ha="center", va="center", fontsize=7.5, color="#ffffff" if v[i, k] > 0.55 else INK)
        ax.set_xticks(range(len(H)), [f"t+{h}" for h in H]); ax.set_title(title, fontsize=9, color=INK)
        ax.grid(False); ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_yticks(range(len(groups)), [groups[i] for i in order])
    fig.supxlabel("Forecast horizon (business days)", fontsize=9, color=INK2, y=0.0)
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("Total share of the gap closed (a + b)", color=INK2); cb.outline.set_visible(False)
    fig.savefig(IMAGE_DIR / "reversion_map.png"); plt.close(fig)


def relmae_by_horizon():
    s = pd.read_csv(RESULTS_DIR / "mr_reference_suite.csv").set_index("model")
    series = [("mean_reversion", "Mean reversion (proposed)", BLUE), ("hw_tft", "TFT", ORANGE),
              ("timesfm_2p5", "TimesFM 2.5 (zero-shot)", AQUA)]
    series = [x for x in series if x[0] in s.index]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.axhline(1.0, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(H[-1], 1.004, "Persistence", ha="right", va="bottom", fontsize=8, color=INK2)
    for key, lab, col in series:
        v = [s.loc[key, f"RelMAE_{h}"] for h in H]
        ax.plot(H, v, color=col, lw=2, marker="o", ms=5, markeredgecolor="#ffffff", markeredgewidth=1, label=lab)
    ax.set_xticks(H, [f"t+{h}" for h in H]); ax.set_xlabel("Forecast horizon (business days)")
    ax.set_ylabel("Relative MAE vs persistence")
    ax.legend(loc="upper right")
    fig.savefig(IMAGE_DIR / "relmae_by_horizon.png"); plt.close(fig)


def group_h250():
    f = pd.read_csv(RESULTS_DIR / "mr_group.csv")
    d = f[f.level == "group"].set_index("cell")[["RelMAE_250", "tft_RelMAE_250"]]
    d = d.rename(columns={"RelMAE_250": "mr", "tft_RelMAE_250": "tft"}).sort_values("mr")
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    y = np.arange(len(d))
    ax.axvline(1.0, color=INK2, lw=1, ls=(0, (4, 3)))
    for i, (a, b) in enumerate(zip(d.mr, d.tft)):
        ax.plot([a, b], [i, i], color=GRID, lw=1.5, zorder=1)
    ax.scatter(d.tft, y, s=36, color=ORANGE, edgecolor="#ffffff", linewidth=1, zorder=3, label="TFT")
    ax.scatter(d.mr, y, s=36, color=BLUE, edgecolor="#ffffff", linewidth=1, zorder=3, label="Mean reversion (proposed)")
    ax.set_yticks(y, list(d.index)); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Relative MAE vs persistence at t+250 (below 1 is better)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))
    fig.savefig(IMAGE_DIR / "group_relmae_h250.png"); plt.close(fig)


if __name__ == "__main__":
    d = error_concentration()
    print(d.round(3).to_string())
    reversion_map(); relmae_by_horizon(); group_h250()
    print("figures written to", IMAGE_DIR)
