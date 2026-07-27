"""Figure 5: the power-law loss weighting decay functions w(h) = 1/h^gamma.

Purely analytic, no data or checkpoint needed. Plots a representative spread
across the full swept grid (gamma in {0.0, ..., 8.0}) and marks gamma = 4.5,
the validation-selected operating point used throughout the paper.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import IMAGE_DIR

OUT = str(IMAGE_DIR / "loss_weighting_curves.png")
SELECTED = 4.5

h = np.arange(1, 251)

# (gamma, descriptive label, colour, linestyle)
CURVES = [
    (0.0, "Unweighted", "#888888", "--"),
    (0.5, "Square-Root", "#1f9e89", "-"),
    (1.0, "Linear", "#31688e", "-"),
    (1.5, "Power 1.5", "#3b528b", "-"),
    (3.0, "Inverse-Cubic", "#443983", "-"),
    (4.5, "Power 4.5", "#d81b60", "-"),
    (8.0, "Extreme", "#fb8c00", "-"),
]


def main():
    plt.rcParams.update({
        "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(8, 5))

    for gamma, label, color, ls in CURVES:
        w = np.ones_like(h, dtype=float) if gamma == 0 else h.astype(float) ** (-gamma)
        is_sel = gamma == SELECTED
        tag = f"gamma = {gamma} ({label})" + ("  [Selected]" if is_sel else "")
        legend = (f"{tag}: w(h) = 1/h^{gamma}" if gamma != 0.0
                  else "gamma = 0.0 (Unweighted): w(h) = 1.0")
        ax.plot(h, w, color=color, linestyle=ls,
                linewidth=2.6 if is_sel else 1.6,
                zorder=5 if is_sel else 2, label=legend)

    ax.set_xlabel("Forecasting Horizon (h steps)")
    ax.set_ylabel("Loss Weight w(h)")
    ax.set_title(r"Comparison of Power-Law Loss Weighting Decay Functions $w(h) = 1/h^\gamma$",
                 fontweight="bold")
    ax.set_ylim(-0.02, 1.03)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print("saved", OUT)


if __name__ == "__main__":
    main()
