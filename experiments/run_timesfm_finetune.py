"""TimesFM 2.5, zero-shot and LoRA fine-tuned, on the matched windows.

Follows Google's official example (timesfm-forecasting/examples/finetuning/
finetune_lora.py): the Transformers checkpoint google/timesfm-2.5-200m-transformers,
PEFT LoRA on all linear layers, random (context, horizon) training windows,
and the model's built-in loss, which is computed in the model's normalized
(RevIN) space and is therefore scale-free.

Protocol, fixed before any score is seen:
  LoRA r = 4, alpha = 8, dropout 0.05, AdamW lr 1e-4 with cosine decay,
  10 epochs of 5,000 random windows, batch 32 (all the example's defaults).
  Context 1,024 business days (the example's 64 suits its weekly retail
  data), horizon 128 steps, the model's native output length.
  training    random windows from each product's history up to the end of 2022
  checkpoint  the epoch with the lowest loss on the last 128 business days of
              each history through 2023 (2023 labels)
  forecasting 256 steps in two autoregressive passes of 128, appending the
              median forecast to the context, for both the zero-shot and the
              fine-tuned model so that the comparison isolates fine-tuning
  selection   zero-shot or LoRA, by overall RelMAE on the 2023 validation windows
  test        the matched 2024-2025 windows, once
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import json, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from paths import CHAMPION_NPZ, EXPERIMENTS_DIR, RESULTS_DIR
from experiments.run_mean_reversion import load

H = [20, 60, 120, 250]
MODEL_ID = "google/timesfm-2.5-200m-transformers"
CONTEXT, STEP, PASSES = 1024, 128, 2
OUT = EXPERIMENTS_DIR / "timesfm_lora"


class RandomWindows(Dataset):
    def __init__(self, series, n, seed=42):
        rng = np.random.default_rng(seed)
        ok = [i for i, s in enumerate(series) if len(s) >= CONTEXT + STEP]
        self.series = series
        self.items = []
        for _ in range(n):
            i = int(rng.choice(ok)); start = int(rng.integers(0, len(series[i]) - CONTEXT - STEP + 1))
            self.items.append((i, start))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        i, s = self.items[k]; x = self.series[i]
        return torch.tensor(x[s:s + CONTEXT], dtype=torch.float32), torch.tensor(x[s + CONTEXT:s + CONTEXT + STEP], dtype=torch.float32)


class LastWindows(Dataset):
    def __init__(self, series):
        self.items = [(torch.tensor(s[-CONTEXT - STEP:-STEP], dtype=torch.float32), torch.tensor(s[-STEP:], dtype=torch.float32))
                      for s in series if len(s) >= CONTEXT + STEP]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        return self.items[k]


def rel_mae(a, f, p, g):
    per = pd.DataFrame({"g": g, "e": np.abs(a - f), "ep": np.abs(a - p)}).groupby("g").mean()
    per = per[per.ep > 0]
    return float(np.exp(np.mean(np.log(np.maximum(per.e, 1e-12) / per.ep))))


def gmean(x):
    return float(np.exp(np.mean(np.log(x))))


@torch.no_grad()
def forecast(model, P, g, o, t0, batch=256):
    """Median forecast at the four horizons, 256 steps in two autoregressive passes."""
    model.eval()
    dev = next(model.parameters()).device
    preds = np.empty((len(g), len(H)))
    for s in range(0, len(g), batch):
        e = min(s + batch, len(g))
        ctx = np.stack([P[g[i], o[i] + 1 - CONTEXT:o[i] + 1] for i in range(s, e)]).astype(np.float32)
        path = []
        for _ in range(PASSES):
            out = model(past_values=torch.tensor(ctx, device=dev), forecast_context_len=CONTEXT, force_flip_invariance=True)
            med = out.mean_predictions[:, :STEP].float().cpu().numpy()
            path.append(med)
            ctx = np.concatenate([ctx, med], axis=1)[:, -CONTEXT:]
        path = np.concatenate(path, axis=1)
        preds[s:e] = path[:, [h - 1 for h in H]]
        if (s // batch) % 80 == 0:
            print(f"    {e:,}/{len(g):,}  elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
    return preds


def windows(z, split):
    g = z[f"{split}_h20_group"].astype(int); o = z[f"{split}_h20_label_idx"].astype(int) - 20
    for h in H:
        assert np.array_equal(z[f"{split}_h{h}_group"].astype(int), g)
        assert np.array_equal(z[f"{split}_h{h}_label_idx"].astype(int) - h, o)
    return g, o


def score(P, g, o, preds, keep):
    rel, mae = [], []
    for k, h in enumerate(H):
        m = keep(o + h)
        a, p, f = P[g[m], o[m] + h], P[g[m], o[m]], preds[m, k]
        rel.append(rel_mae(a, f, p, g[m])); mae.append(float(np.mean(np.abs(a - f))))
    return gmean(rel), float(np.mean(mae)), rel


def main():
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import TimesFm2_5ModelForPrediction
    t0 = time.time()
    torch.manual_seed(42)
    P, val_start, test_start, codes, group, family, level = load()
    z = np.load(CHAMPION_NPZ)
    gv, ov = windows(z, "val"); gt, ot = windows(z, "test")
    assert ov.min() + 1 >= CONTEXT and ot.min() + 1 >= CONTEXT
    OUT.mkdir(parents=True, exist_ok=True)

    def base_model():
        return TimesFm2_5ModelForPrediction.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")

    # ---- LoRA fine-tuning, the official example's defaults
    model = get_peft_model(base_model(), LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", lora_dropout=0.05, bias="none"))
    model.print_trainable_parameters()
    train = DataLoader(RandomWindows([P[i, :val_start].astype(np.float32) for i in range(len(codes))], 5000),
                       batch_size=32, shuffle=True, drop_last=True)
    valid = DataLoader(LastWindows([P[i, :test_start].astype(np.float32) for i in range(len(codes))]), batch_size=32)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10 * len(train))
    best = float("inf")
    for epoch in range(1, 11):
        model.train(); tl = []
        for x, y in train:
            loss = model(past_values=x.cuda(), future_values=y.cuda(), forecast_context_len=CONTEXT).loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(); sched.step(); tl.append(loss.item())
        model.eval(); vl = []
        with torch.no_grad():
            for x, y in valid:
                vl.append(model(past_values=x.cuda(), future_values=y.cuda(), forecast_context_len=CONTEXT).loss.item())
        v = float(np.mean(vl))
        print(f"epoch {epoch}: train loss {np.mean(tl):.4f}  val loss {v:.4f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if v < best:
            best = v; model.save_pretrained(OUT)
    print(f"best validation loss {best:.4f}", flush=True)
    del model; torch.cuda.empty_cache()

    models = {"zero_shot": base_model(), "lora": PeftModel.from_pretrained(base_model(), OUT)}
    val_rows = {}
    for name, m in models.items():
        print(f"validation forecasts: {name}", flush=True)
        pv = forecast(m, P, gv, ov, t0)
        r, mae, byh = score(P, gv, ov, pv, lambda li: li >= val_start)
        val_rows[name] = dict(val_RelMAE=r, val_MAE=mae, **{f"val_RelMAE_{h}": x for h, x in zip(H, byh)})
        print(f"  {name}: validation RelMAE {r:.4f}  MAE {mae:.2f}", flush=True)
    selected = min(val_rows, key=lambda k: val_rows[k]["val_RelMAE"])
    print(f"validation-selected: {selected}", flush=True)

    rows = []
    for name in (["zero_shot", "lora"] if selected == "lora" else ["zero_shot"]):
        print(f"test forecasts: {name}", flush=True)
        pt = forecast(models[name], P, gt, ot, t0)
        out = {}
        for k, h in enumerate(H):
            out.update({f"test_h{h}_group": gt, f"test_h{h}_label_idx": ot + h, f"test_h{h}_actual": P[gt, ot + h],
                        f"test_h{h}_persistence": P[gt, ot], f"test_h{h}_median": pt[:, k]})
        tag = "lora" if name == "lora" else "ar1024_zeroshot"
        np.savez_compressed(EXPERIMENTS_DIR / f"timesfm_{tag}_predictions.npz", **out)
        r, mae, byh = score(P, gt, ot, pt, lambda li: np.ones(len(li), bool))
        rows.append(dict(model=f"timesfm_{tag}", test_RelMAE=r, test_MAE=mae, **{f"test_RelMAE_{h}": x for h, x in zip(H, byh)}, **val_rows[name]))
        print(f"  {name}: TEST RelMAE {r:.4f}  MAE {mae:.2f}  by horizon {np.round(byh, 3).tolist()}", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "timesfm_finetune_results.csv", index=False)
    json.dump({"validation": val_rows, "selected": selected, "best_val_loss": best},
              open(RESULTS_DIR / "timesfm_finetune_selection.json", "w"), indent=1)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
