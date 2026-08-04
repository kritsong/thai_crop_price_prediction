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

pd.DataFrame(rows).to_csv(OUT, index=False)
print("saved", OUT)
