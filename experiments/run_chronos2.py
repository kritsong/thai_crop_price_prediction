"""Chronos-2 zero-shot and fine-tuned, on the matched windows.

Protocol, fixed before any score is seen:
  candidates  zero-shot; full fine-tuning (learning rate 1e-6, the library
              default); LoRA fine-tuning (learning rate 1e-5, the library's
              recommendation for LoRA). Both fine-tunes run 1,000 steps of 64 series
              with a 256-step horizon and a 2,048-step context (the library default
              batch of 256 exhausts the 24 GB GPU; changed before any score was seen).
  training    each product's cleaned history up to the end of 2022, so no
              label from 2023 or later is seen
  checkpoint  the library evaluates every 100 steps on the last 256 business
              days of each history through 2023 (the 2023 labels) and keeps
              the step with the lowest validation loss
  selection   the candidate with the lowest overall RelMAE on the 2023
              validation windows of the evaluation harness
  test        the matched 2024-2025 windows; the paper reports the zero-shot
              model and the validation-selected fine-tuned candidate
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import json, time
import numpy as np
import pandas as pd
import torch

from paths import CHAMPION_NPZ, EXPERIMENTS_DIR, RESULTS_DIR
from experiments.run_mean_reversion import load

H = [20, 60, 120, 250]
HORIZON = 256
CONTEXT = 2048
OUT = EXPERIMENTS_DIR / "chronos2"


def rel_mae(a, f, p, g):
    per = pd.DataFrame({"g": g, "e": np.abs(a - f), "ep": np.abs(a - p)}).groupby("g").mean()
    per = per[per.ep > 0]
    return float(np.exp(np.mean(np.log(np.maximum(per.e, 1e-12) / per.ep))))


def gmean(x):
    return float(np.exp(np.mean(np.log(x))))


def forecast(pipe, P, g, o, t0):
    """Median forecast at the four horizons for windows with origins o (unique windows, shared by horizons)."""
    preds = np.empty((len(g), len(H)))
    step = 4096
    for s in range(0, len(g), step):
        e = min(s + step, len(g))
        ctx = [torch.tensor(P[g[i], max(0, o[i] + 1 - CONTEXT):o[i] + 1], dtype=torch.float32) for i in range(s, e)]
        _, med = pipe.predict_quantiles(ctx, prediction_length=HORIZON, batch_size=256)
        m = np.stack([x[0].float().cpu().numpy() for x in med])          # (n, 256) median
        preds[s:e] = m[:, [h - 1 for h in H]]
        if (s // step) % 5 == 0:
            print(f"    {e:,}/{len(g):,}  elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
    return preds


def windows(z, split):
    g = z[f"{split}_h20_group"].astype(int); o = z[f"{split}_h20_label_idx"].astype(int) - 20
    for h in H:   # the windows are shared by all horizons, only the label filter differs
        assert np.array_equal(z[f"{split}_h{h}_group"].astype(int), g)
        assert np.array_equal(z[f"{split}_h{h}_label_idx"].astype(int) - h, o)
    return g, o


def score(P, g, o, preds, keep_fn):
    rel, mae = [], []
    for k, h in enumerate(H):
        m = keep_fn(o + h)
        a, p, f = P[g[m], o[m] + h], P[g[m], o[m]], preds[m, k]
        rel.append(rel_mae(a, f, p, g[m])); mae.append(float(np.mean(np.abs(a - f))))
    return gmean(rel), float(np.mean(mae)), rel


def main():
    from chronos import BaseChronosPipeline
    t0 = time.time()
    P, val_start, test_start, codes, group, family, level = load()
    z = np.load(CHAMPION_NPZ)
    gv, ov = windows(z, "val"); gt, ot = windows(z, "test")
    for h in H:
        assert np.allclose(P[gt, ot + h], z[f"test_h{h}_actual"], atol=1e-3)
    OUT.mkdir(parents=True, exist_ok=True)

    base = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
    print(f"Chronos-2 loaded ({time.time() - t0:.0f}s)", flush=True)

    train_inputs = [P[i, :val_start].astype(np.float32) for i in range(len(codes))]      # labels through 2022
    valid_inputs = [P[i, :test_start].astype(np.float32) for i in range(len(codes))]    # last 256 days are 2023

    pipes = {"zero_shot": base}
    for mode, lr in [("full", 1e-6), ("lora", 1e-5)]:
        print(f"fine-tuning {mode} (lr {lr})", flush=True)
        pipes[f"finetune_{mode}"] = base.fit(
            inputs=train_inputs, prediction_length=HORIZON, validation_inputs=valid_inputs,
            finetune_mode=mode, context_length=CONTEXT, learning_rate=lr, num_steps=1000, batch_size=64,
            output_dir=OUT / mode, remove_printer_callback=True)
        print(f"  done ({(time.time() - t0) / 60:.1f} min)", flush=True)

    val_rows = {}
    for name, pipe in pipes.items():
        print(f"validation forecasts: {name}", flush=True)
        pv = forecast(pipe, P, gv, ov, t0)
        r, m, byh = score(P, gv, ov, pv, lambda li: li >= val_start)
        val_rows[name] = dict(val_RelMAE=r, val_MAE=m, **{f"val_RelMAE_{h}": x for h, x in zip(H, byh)})
        print(f"  {name}: validation RelMAE {r:.4f}  MAE {m:.2f}", flush=True)

    finetuned = {k: v for k, v in val_rows.items() if k != "zero_shot"}
    selected = min(finetuned, key=lambda k: finetuned[k]["val_RelMAE"])
    print(f"validation-selected fine-tuned candidate: {selected}", flush=True)

    rows = []
    for name in ["zero_shot", selected]:
        print(f"test forecasts: {name}", flush=True)
        pt = forecast(pipes[name], P, gt, ot, t0)
        out = {}
        for k, h in enumerate(H):
            out.update({f"test_h{h}_group": gt, f"test_h{h}_label_idx": ot + h, f"test_h{h}_actual": P[gt, ot + h],
                        f"test_h{h}_persistence": P[gt, ot], f"test_h{h}_median": pt[:, k]})
        tag = "zeroshot" if name == "zero_shot" else "finetuned"
        np.savez_compressed(EXPERIMENTS_DIR / f"chronos2_{tag}_predictions.npz", **out)
        r, m, byh = score(P, gt, ot, pt, lambda li: np.ones(len(li), bool))
        rows.append(dict(model=f"chronos2_{tag}", candidate=name, test_RelMAE=r, test_MAE=m,
                         **{f"test_RelMAE_{h}": x for h, x in zip(H, byh)}, **val_rows[name]))
        print(f"  {name}: TEST RelMAE {r:.4f}  MAE {m:.2f}  by horizon {np.round(byh, 3).tolist()}", flush=True)

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "chronos2_results.csv", index=False)
    json.dump({"validation": val_rows, "selected_finetuned": selected}, open(RESULTS_DIR / "chronos2_selection.json", "w"), indent=1)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
