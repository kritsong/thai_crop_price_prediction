"""Diebold-Mariano-type significance tests for the selected model vs the
matched Lag-1 baseline, computed from the saved per-window test predictions.

Method. The classical time-series DM test is degenerate here because h-step
forecasts from consecutive origins overlap. At h = 250 the 273 windows contain
roughly one effective independent observation, so no HAC correction can rescue
a time-wise test. We therefore cluster by product: for each of the 404 crops
we collapse its windows to one mean absolute-error differential
d_p = mean(|e_model|) - mean(|e_baseline|), then test the cross-section of
d_p values (negative = model better). Cross-product dependence is weak in this
data (mean pairwise return correlation 0.012, Section 3.6 of the paper), so
the product-level cross-section is close to independent. We report a two-sided
t-test and a Wilcoxon signed-rank test per horizon and overall.
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
