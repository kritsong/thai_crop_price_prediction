"""TimesFM, zero-shot, on the matched test windows.

Each window gets the product's full cleaned price history up to the forecast
origin (TimesFM accepts long contexts, so nothing is truncated), and the model
forecasts 256 steps. The median of the predictive distribution is scored at
20, 60, 120 and 250 business days against the same actuals as every other
model. No fine-tuning and no covariates, the same protocol as Chronos-Bolt.

  python experiments/run_timesfm_zeroshot.py --version 2.5
  python experiments/run_timesfm_zeroshot.py --version 3.0
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse, time
import numpy as np
import pandas as pd
import torch

from paths import CHAMPION_NPZ, EXPERIMENTS_DIR, RESULTS_DIR
from experiments.run_mean_reversion import load

H = [20, 60, 120, 250]
HORIZON = 256
MAX_CONTEXT = 2048


def build_model(version):
    if version == "2.5":
        import timesfm
        torch.set_float32_matmul_precision("high")
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        model.compile(timesfm.ForecastConfig(
            max_context=MAX_CONTEXT, max_horizon=HORIZON, normalize_inputs=True,
            use_continuous_quantile_head=True, force_flip_invariance=True,
            infer_is_positive=True, fix_quantile_crossing=True, per_core_batch_size=256))

        def forecast(contexts):
            point, _ = model.forecast(horizon=HORIZON, inputs=contexts)
            return np.asarray(point)
        return forecast
    if version == "3.0":
        import timesfm   # timesfm >= 3.0
        # the library default inference batch is 4 series, far too small for 110k windows
        ev = timesfm.TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", device="cuda", per_core_batch_size=256)

        def forecast(contexts):
            # .forecast is the 0.5 decile, the same median point forecast used for 2.5 and Chronos-Bolt
            out = ev.predict_batch(contexts=contexts, horizon=HORIZON, make_positive=True)
            return np.stack([np.asarray(o.forecast) for o in out])
        return forecast
    raise ValueError(version)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["2.5", "3.0"], required=True)
    ap.add_argument("--batch", type=int, default=1024)
    args = ap.parse_args()
    t0 = time.time()

    P, val_start, test_start, codes, group, family, level = load()
    z = np.load(CHAMPION_NPZ)
    # the matched windows share origins across horizons, so forecast each origin once
    g = z["test_h20_group"].astype(int); o = z["test_h20_label_idx"].astype(int) - 20
    for h in H:
        assert np.array_equal(z[f"test_h{h}_group"].astype(int), g)
        assert np.array_equal(z[f"test_h{h}_label_idx"].astype(int) - h, o)
        assert np.allclose(P[g, o + h], z[f"test_h{h}_actual"], atol=1e-3)
        assert np.allclose(P[g, o], z[f"test_h{h}_persistence"], atol=1e-3)
    print(f"{len(g):,} matched windows, context lengths {o.min() + 1}-{o.max() + 1}", flush=True)

    forecast = build_model(args.version)
    print(f"model loaded ({time.time() - t0:.0f}s)", flush=True)

    preds = np.empty((len(g), len(H)))
    idx = [h - 1 for h in H]
    for s in range(0, len(g), args.batch):
        e = min(s + args.batch, len(g))
        ctx = [P[g[i], max(0, o[i] + 1 - MAX_CONTEXT):o[i] + 1].astype(np.float32) for i in range(s, e)]
        f = forecast(ctx)
        preds[s:e] = f[:, idx]
        if (s // args.batch) % 20 == 0:
            print(f"  {e:,}/{len(g):,}  elapsed {(time.time() - t0) / 60:.1f} min", flush=True)

    tag = args.version.replace(".", "p")
    out, row = {}, {"model": f"timesfm_{tag}_zeroshot"}
    for k, h in enumerate(H):
        a = P[g, o + h]
        out.update({f"test_h{h}_group": g, f"test_h{h}_label_idx": o + h, f"test_h{h}_actual": a,
                    f"test_h{h}_persistence": P[g, o], f"test_h{h}_median": preds[:, k]})
        row[f"mae_{h}"] = float(np.mean(np.abs(a - preds[:, k])))
        print(f"TimesFM {args.version} h{h}: MAE {row[f'mae_{h}']:.2f}  (persistence {np.mean(np.abs(a - P[g, o])):.2f})")
    row["mae_overall"] = float(np.mean([row[f"mae_{h}"] for h in H]))
    print(f"TimesFM {args.version} overall MAE {row['mae_overall']:.2f}  ({(time.time() - t0) / 60:.1f} min)")
    np.savez_compressed(EXPERIMENTS_DIR / f"timesfm_{tag}_zeroshot_predictions.npz", **out)
    pd.DataFrame([row]).to_csv(RESULTS_DIR / f"timesfm_{tag}_zeroshot.csv", index=False)


if __name__ == "__main__":
    main()
