"""Figure 6: horizon-specific and overall test MAE against the decay exponent.

Reads the swept results from results/gamma_sweep_summary.csv, so it reproduces
the published figure without retraining anything.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import RESULTS_DIR, IMAGE_DIR

SRC = RESULTS_DIR / "gamma_sweep_summary.csv"
OUT = str(IMAGE_DIR / "mae_vs_gamma.png")
HORIZONS = [20, 60, 120, 250]
SELECTED = 4.5
# Table 6 and this figure report ONE training run per exponent. The sweep file
# also holds replicate runs at gamma = 3.5, 4.0 and 4.5; averaging them would
# silently disagree with the published table (39.76 vs 41.89 THB at gamma=4.5),
# so the primary run is selected explicitly.
PRIMARY_RUN = 42

COLORS = {20: "#1f9e89", 60: "#31688e", 120: "#443983", 250: "#d81b60"}


def main():
    df = pd.read_csv(SRC)
    df = df[df["Horizon"].isin(HORIZONS) & (df["Seed"] == PRIMARY_RUN)]

    piv = df.pivot_table(index="Gamma", columns="Horizon", values="MAE").sort_index()
    assert not piv.isna().any().any(), "missing (gamma, horizon) cells for the primary run"
    overall = piv[HORIZONS].mean(axis=1)

    published = 39.761308            # Table 7 overall MAE of the selected model
    got = float(overall.loc[SELECTED])
    assert abs(got - published) < 0.01, f"gamma={SELECTED} overall {got:.2f} != {published:.2f}"

    plt.rcParams.update({
        "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    for h in HORIZONS:
        ax.plot(piv.index, piv[h], marker="o", markersize=3.5, linewidth=1.4,
                color=COLORS[h], label=f"$t+{h}$")
    ax.plot(overall.index, overall.values, marker="s", markersize=4.5,
            linewidth=2.4, color="#111111", label="Overall (mean of four horizons)",
            zorder=5)

    ax.axvline(SELECTED, color="#666666", linestyle=":", linewidth=1.1)
    best = float(overall.loc[SELECTED])
    ax.annotate(f"Selected model\ngamma = {SELECTED}, {best:.2f} THB",
                xy=(SELECTED, best), xytext=(SELECTED + 0.55, best + 9),
                fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color="#444444", linewidth=0.9))

    ax.set_xlabel(r"Decay exponent $\gamma$")
    ax.set_ylabel("Test MAE (THB)")
    ax.set_title("Out-of-sample error against the loss decay exponent",
                 fontweight="bold")
    ax.legend(fontsize=8.5, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print("saved", OUT)
    print(f"overall minimum at gamma={overall.idxmin()} ({overall.min():.2f} THB); "
          f"selected gamma={SELECTED} ({best:.2f} THB)")


if __name__ == "__main__":
    main()
