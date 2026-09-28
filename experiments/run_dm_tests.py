"""Significance tests, selected model against the Lag-1 baseline.

A plain Diebold-Mariano test does not work here. Forecasts from consecutive
origins overlap, and at h=250 the 273 windows per product are worth about one
independent observation, which no HAC correction fixes. So we cluster by
product instead: collapse each crop to one mean error differential, then test
those 404 numbers. Negative means our model wins. Returns are close to
uncorrelated across products, so treating them as independent is reasonable.

Reports a t-test and a Wilcoxon per horizon.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
from scipy import stats

# make the repository root importable regardless of how this is invoked
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import CHAMPION_NPZ, RESULTS_DIR

NPZ = str(CHAMPION_NPZ)
OUT = str(RESULTS_DIR / "dm_tests.csv")
H = [20, 60, 120, 250]

z = np.load(NPZ)
rows = []
per_prod = {}

for h in H:
    act = z[f"test_h{h}_actual"]
    med = z[f"test_h{h}_median"]
    per = z[f"test_h{h}_persistence"]
    grp = z[f"test_h{h}_group"].astype(int)
    d = np.abs(act - med) - np.abs(act - per)
    df = pd.DataFrame({"g": grp, "d": d})
    dp = df.groupby("g")["d"].mean()
    per_prod[h] = dp
    t, p_t = stats.ttest_1samp(dp.values, 0.0)
    w, p_w = stats.wilcoxon(dp.values)
    rows.append({
        "horizon": h, "n_products": len(dp), "mean_diff_thb": float(dp.mean()),
        "share_negative": float((dp < 0).mean()),
        "t_stat": float(t), "p_ttest": float(p_t),
        "wilcoxon_stat": float(w), "p_wilcoxon": float(p_w),
    })
    print(f"h={h}: mean d={dp.mean():+8.2f} THB  share better={100*(dp<0).mean():5.1f}%  "
          f"t={t:+7.2f} p={p_t:.2e}  wilcoxon p={p_w:.2e}")

overall = pd.concat(per_prod, axis=1).mean(axis=1)
t, p_t = stats.ttest_1samp(overall.values, 0.0)
w, p_w = stats.wilcoxon(overall.values)
rows.append({
    "horizon": 0, "n_products": len(overall), "mean_diff_thb": float(overall.mean()),
    "share_negative": float((overall < 0).mean()),
    "t_stat": float(t), "p_ttest": float(p_t),
    "wilcoxon_stat": float(w), "p_wilcoxon": float(p_w),
})
print(f"overall: mean d={overall.mean():+8.2f} THB  share better={100*(overall<0).mean():5.1f}%  "
      f"t={t:+7.2f} p={p_t:.2e}  wilcoxon p={p_w:.2e}")

def holm_adjust(p_values):
    """Holm-Bonferroni step-down adjustment, applied across the four horizons."""
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running_max = 0.0
    m = len(values)
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * values[idx])
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


out = pd.DataFrame(rows)

# The paper reports the four horizon comparisons as a family, so the horizon
# rows are corrected together. The overall row is a separate summary and is
# left unadjusted.
mask = out["horizon"].isin(H)
for src, dst in [("p_ttest", "p_ttest_holm"), ("p_wilcoxon", "p_wilcoxon_holm")]:
    out[dst] = np.nan
    out.loc[mask, dst] = holm_adjust(out.loc[mask, src].to_numpy())

print("\nHolm-adjusted across the four horizons:")
for _, r in out[mask].iterrows():
    print(f"  h={int(r.horizon):3d}  t-test {r.p_ttest:.2e} -> {r.p_ttest_holm:.2e}"
          f"   wilcoxon {r.p_wilcoxon:.2e} -> {r.p_wilcoxon_holm:.2e}")

out.to_csv(OUT, index=False)
print("saved", OUT)
