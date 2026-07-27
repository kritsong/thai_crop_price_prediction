"""Fair, validation-based hyperparameter search for the learned global reference
models (LightGBM, MLP, LSTM, Transformer) reported in Table 5.

Protocol (mirrors Section 4.2 of the paper):
  * Search phase: each candidate configuration is fitted on labels within
    2018-2022 only and scored by matched-window MAE on validation windows whose
    horizon-h label falls inside 2023, averaged over h in {20, 60, 120, 250}.
  * Final phase: the validation-selected configuration (and, for reference, the
    published configuration) is refitted on labels through 2023, the same
    training window the published reference models used, and evaluated on the
    identical 2024-2025 matched test windows as Table 5.
  * Equal light budget: 10 candidate configurations per model class, matching
    the 10-trial budget of the TFT architecture search (src/models/tune_tft.py).
    The published configuration is always one of the 10 candidates.

Outputs results/reference_hparam_search.csv (per-configuration validation MAE)
and results/reference_hparam_final.csv (test MAE of selected vs published).
"""
import os
import sys, io, json, glob, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from joblib import Parallel, delayed

from src.data.loader import CropDataLoader, EmptySeriesError
from src.features.generator import FeatureGenerator
from src.models.train import GlobalMLP, GlobalLSTM, GlobalTransformer, RobustLabelEncoder

HORIZONS = [1, 20, 60, 120, 250]
EVAL_H = [20, 60, 120, 250]
from paths import DATA_GLOB, RESULTS_DIR as _RESULTS_DIR

RESULTS_DIR = str(_RESULTS_DIR)
UNIVERSE = set(pd.read_csv(os.path.join(RESULTS_DIR, "metrics_by_horizon.csv"))["product_id"].unique())
print(f"Evaluation universe: {len(UNIVERSE)} products (matching Table 5)")


def build_product(fp, features):
    pid = os.path.basename(fp).split(".")[0]
    if pid not in UNIVERSE:
        return None
    loader = CropDataLoader()
    gen = FeatureGenerator()
    try:
        df_aligned = loader.load_and_preprocess(fp)
        if len(df_aligned) < 70:
            return None
        df_feat = gen.generate_features(df_aligned)
        if len(df_feat) == 0:
            return None
    except (json.JSONDecodeError, EmptySeriesError, KeyError, Exception):
        return None

    # full-path targets and label dates for per-horizon validation evaluation
    full = df_feat.copy()
    dates = pd.to_datetime(full["date"]).reset_index(drop=True)
    full = full.reset_index(drop=True)
    for h in HORIZONS:
        full[f"target_{h}"] = full["price_min"].shift(-(h - 1))
        full[f"label_date_{h}"] = dates.shift(-(h - 1))

    tr22, val23, te = gen.split_train_val_test(df_feat)     # labels within 2018-22 / 2023 / 2024-25
    tr23, _ = gen.split_train_test(df_feat)                 # labels within 2018-2023 (published training window)
    return {"pid": pid, "full": full, "tr22": tr22, "te": te, "tr23": tr23}


def main():
    t0 = time.time()
    torch.manual_seed(42)
    np.random.seed(42)
    gen = FeatureGenerator()
    features = gen.features
    filepaths = glob.glob(DATA_GLOB)
    print(f"Scanning {len(filepaths)} files...")
    parts = Parallel(n_jobs=-1)(delayed(build_product)(fp, features) for fp in filepaths)
    parts = [p for p in parts if p is not None]
    print(f"Built feature frames for {len(parts)} products in {time.time()-t0:.0f}s")
    assert len(parts) == len(UNIVERSE), f"universe mismatch: {len(parts)} vs {len(UNIVERSE)}"

    full = pd.concat([p["full"] for p in parts], ignore_index=True)
    tr22 = pd.concat([p["tr22"] for p in parts], ignore_index=True)
    te = pd.concat([p["te"] for p in parts], ignore_index=True)
    tr23 = pd.concat([p["tr23"] for p in parts], ignore_index=True)

    # per-horizon validation windows: label falls inside 2023, target defined
    val_h = {}
    for h in EVAL_H:
        ld = pd.to_datetime(full[f"label_date_{h}"])
        m = (ld >= "2023-01-01") & (ld <= "2023-12-31") & full[f"target_{h}"].notna()
        val_h[h] = full[m].copy()
        print(f"validation windows h={h}: {m.sum():,}")
    print(f"test windows per horizon: {len(te):,}")

    # encoders and encoded frames (fit on the widest training window; product
    # sets are identical across splits so mappings coincide)
    enc_p, enc_c, enc_g = RobustLabelEncoder(), RobustLabelEncoder(), RobustLabelEncoder()
    enc_p.fit(tr23["product_id"]); enc_c.fit(tr23["category_name"]); enc_g.fit(tr23["group_name"])
    for df in [tr22, tr23, te] + list(val_h.values()):
        df["prod_enc"] = enc_p.transform(df["product_id"])
        df["cat_enc"] = enc_c.transform(df["category_name"])
        df["group_enc"] = enc_g.transform(df["group_name"])

    gfeat = features + ["prod_enc", "cat_enc", "group_enc"]
    catf = ["prod_enc", "cat_enc", "group_enc", "month", "day_of_week"]
    cat_idx = [gfeat.index(c) for c in catf]
    contf = [f for f in gfeat if f not in catf]
    num_prods = int(tr23["prod_enc"].max()); num_cats = int(tr23["cat_enc"].max()); num_groups = int(tr23["group_enc"].max())

    baseline_val = np.mean([np.mean(np.abs(val_h[h]["target_" + str(h)] - val_h[h]["lag_1"])) for h in EVAL_H])
    baseline_test = np.mean([np.mean(np.abs(te[f"target_{h}"] - te["lag_1"])) for h in EVAL_H])
    print(f"persistence MAE  validation {baseline_val:.2f}  test {baseline_test:.2f}")

    search_rows, final_rows = [], []

    # ---------------- LightGBM ----------------
    LGB = [
        {"num_leaves": 2633, "max_depth": 8, "min_data_in_leaf": 700, "learning_rate": 0.05, "feature_fraction": 0.99, "bagging_fraction": 0.74, "bagging_freq": 3},  # published
        {"num_leaves": 31, "max_depth": -1, "min_data_in_leaf": 20, "learning_rate": 0.05, "feature_fraction": 1.0, "bagging_fraction": 1.0, "bagging_freq": 0},
        {"num_leaves": 63, "max_depth": -1, "min_data_in_leaf": 20, "learning_rate": 0.1, "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1},
        {"num_leaves": 127, "max_depth": 8, "min_data_in_leaf": 100, "learning_rate": 0.05, "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1},
        {"num_leaves": 255, "max_depth": 10, "min_data_in_leaf": 100, "learning_rate": 0.05, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1},
        {"num_leaves": 511, "max_depth": 10, "min_data_in_leaf": 300, "learning_rate": 0.05, "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1},
        {"num_leaves": 1023, "max_depth": 12, "min_data_in_leaf": 300, "learning_rate": 0.03, "feature_fraction": 0.9, "bagging_fraction": 0.7, "bagging_freq": 3},
        {"num_leaves": 2633, "max_depth": 8, "min_data_in_leaf": 700, "learning_rate": 0.1, "feature_fraction": 0.99, "bagging_fraction": 0.74, "bagging_freq": 3},
        {"num_leaves": 4095, "max_depth": 12, "min_data_in_leaf": 1000, "learning_rate": 0.05, "feature_fraction": 0.8, "bagging_fraction": 0.7, "bagging_freq": 3},
        {"num_leaves": 255, "max_depth": -1, "min_data_in_leaf": 50, "learning_rate": 0.1, "feature_fraction": 1.0, "bagging_fraction": 0.9, "bagging_freq": 1},
    ]

    def lgb_fit_eval(train_df, cfg, eval_frames):
        """train per-horizon boosters on train_df, return per-horizon MAE on eval_frames[h]"""
        X = train_df[gfeat].values
        maes = {}
        for h in EVAL_H:
            params = {"objective": "regression", "metric": "rmse", "verbosity": -1, "seed": 42, **cfg}
            ds = lgb.Dataset(X, label=train_df[f"target_{h}"].values, categorical_feature=cat_idx)
            booster = lgb.train(params, ds, num_boost_round=100)
            ev = eval_frames[h]
            pred = booster.predict(ev[gfeat].values)
            pred = np.where(np.isfinite(pred), pred, ev["lag_1"].values)
            maes[h] = float(np.mean(np.abs(ev[f"target_{h}"].values - pred)))
        return maes

    print("\n=== LightGBM search (10 configs) ===")
    lgb_val = []
    for i, cfg in enumerate(LGB):
        maes = lgb_fit_eval(tr22, cfg, val_h)
        mean_mae = float(np.mean(list(maes.values())))
        lgb_val.append(mean_mae)
        search_rows.append({"family": "lightgbm", "config_id": i, "published": i == 0,
                            "params": json.dumps(cfg), **{f"val_mae_{h}": maes[h] for h in EVAL_H},
                            "val_mae_mean": mean_mae})
        print(f"  cfg{i}{' (published)' if i == 0 else ''}: val MAE {mean_mae:.2f}  {maes}")
    best_i = int(np.argmin(lgb_val))
    print(f"  selected: cfg{best_i}")
    for label, cfg in [("selected", LGB[best_i]), ("published", LGB[0])]:
        if label == "published" and best_i == 0:
            continue
        maes = lgb_fit_eval(tr23, cfg, {h: te for h in EVAL_H})
        final_rows.append({"family": "lightgbm", "which": label, "config_id": LGB.index(cfg),
                           "params": json.dumps(cfg), **{f"test_mae_{h}": maes[h] for h in EVAL_H},
                           "test_mae_mean": float(np.mean(list(maes.values())))})
        print(f"  {label} test MAE {np.mean(list(maes.values())):.2f}  {maes}")
    if best_i == 0:
        r = dict(final_rows[-1]); r["which"] = "published"; final_rows.append(r)

    # ---------------- neural helpers ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nneural device: {device}")

    def make_tensors(train_df):
        sx, sy = StandardScaler(), StandardScaler()
        Xc = sx.fit_transform(train_df[contf].values)
        Y = sy.fit_transform(train_df[[f"target_{h}" for h in HORIZONS]].values)
        Xcat = train_df[catf].values
        return sx, sy, (torch.tensor(Xc, dtype=torch.float32), torch.tensor(Xcat, dtype=torch.long), torch.tensor(Y, dtype=torch.float32))

    def train_net(model, tensors, lr, epochs, seq):
        Xc, Xcat, Y = tensors
        if seq:
            Xc, Xcat = Xc.unsqueeze(1), Xcat.unsqueeze(1)
        ds = TensorDataset(Xc.to(device), Xcat.to(device), Y.to(device))
        dl = DataLoader(ds, batch_size=4096, shuffle=True)
        opt = optim.Adam(model.parameters(), lr=lr)
        crit = nn.MSELoss()
        model.train()
        for _ in range(epochs):
            for bc, bk, by in dl:
                opt.zero_grad()
                loss = crit(model(bc, bk), by)
                loss.backward()
                opt.step()
        return model

    def net_eval(model, sx, sy, eval_frames, seq):
        model.eval()
        maes = {}
        for h in EVAL_H:
            ev = eval_frames[h]
            Xc = torch.tensor(sx.transform(ev[contf].values), dtype=torch.float32)
            Xcat = torch.tensor(ev[catf].values, dtype=torch.long)
            if seq:
                Xc, Xcat = Xc.unsqueeze(1), Xcat.unsqueeze(1)
            chunks = []
            for i0 in range(0, len(Xc), 8192):
                with torch.no_grad():
                    chunks.append(model(Xc[i0:i0 + 8192].to(device), Xcat[i0:i0 + 8192].to(device)).cpu())
            pred = torch.cat(chunks).numpy()
            pred = sy.inverse_transform(pred)[:, HORIZONS.index(h)]
            pred = np.where(np.isfinite(pred), pred, ev["lag_1"].values)
            maes[h] = float(np.mean(np.abs(ev[f"target_{h}"].values - pred)))
        return maes

    class MLP2(nn.Module):
        """GlobalMLP with configurable widths/dropout (same embeddings)."""
        def __init__(self, num_cont, h1, h2, drop):
            super().__init__()
            self.prod_emb = nn.Embedding(num_prods + 1, 16)
            self.cat_emb = nn.Embedding(num_cats + 1, 8)
            self.group_emb = nn.Embedding(num_groups + 1, 8)
            self.month_emb = nn.Embedding(13, 4)
            self.day_emb = nn.Embedding(7, 4)
            self.net = nn.Sequential(
                nn.Linear(num_cont + 40, h1), nn.ReLU(), nn.Dropout(drop),
                nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(drop),
                nn.Linear(h2, len(HORIZONS)))
        def forward(self, xc, xk):
            e = [self.prod_emb(xk[:, 0]), self.cat_emb(xk[:, 1]), self.group_emb(xk[:, 2]),
                 self.month_emb(xk[:, 3]), self.day_emb(xk[:, 4])]
            return self.net(torch.cat([xc] + e, dim=-1))

    class LSTM2(nn.Module):
        def __init__(self, num_cont, hidden):
            super().__init__()
            self.prod_emb = nn.Embedding(num_prods + 1, 16)
            self.cat_emb = nn.Embedding(num_cats + 1, 8)
            self.group_emb = nn.Embedding(num_groups + 1, 8)
            self.month_emb = nn.Embedding(13, 4)
            self.day_emb = nn.Embedding(7, 4)
            self.lstm = nn.LSTM(num_cont + 40, hidden, batch_first=True)
            self.fc = nn.Linear(hidden, len(HORIZONS))
        def forward(self, xc, xk):
            e = [self.prod_emb(xk[:, :, 0]), self.cat_emb(xk[:, :, 1]), self.group_emb(xk[:, :, 2]),
                 self.month_emb(xk[:, :, 3]), self.day_emb(xk[:, :, 4])]
            out, _ = self.lstm(torch.cat([xc] + e, dim=-1))
            return self.fc(out[:, -1, :])

    class TF2(nn.Module):
        def __init__(self, num_cont, d_model, nhead, layers):
            super().__init__()
            self.prod_emb = nn.Embedding(num_prods + 1, d_model)
            self.cat_emb = nn.Embedding(num_cats + 1, d_model)
            self.group_emb = nn.Embedding(num_groups + 1, d_model)
            self.month_emb = nn.Embedding(13, d_model)
            self.day_emb = nn.Embedding(7, d_model)
            self.cont_proj = nn.Linear(num_cont, d_model)
            enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True,
                                             dim_feedforward=d_model * 4, dropout=0.1)
            self.tr = nn.TransformerEncoder(enc, num_layers=layers)
            self.fc = nn.Linear(6 * d_model, len(HORIZONS))
        def forward(self, xc, xk):
            toks = [self.cont_proj(xc).unsqueeze(1),
                    self.prod_emb(xk[:, 0]).unsqueeze(1), self.cat_emb(xk[:, 1]).unsqueeze(1),
                    self.group_emb(xk[:, 2]).unsqueeze(1), self.month_emb(xk[:, 3]).unsqueeze(1),
                    self.day_emb(xk[:, 4]).unsqueeze(1)]
            out = self.tr(torch.cat(toks, dim=1))
            return self.fc(out.reshape(out.size(0), -1))

    NET_FAMILIES = {
        "mlp": {
            "configs": [
                {"h1": 128, "h2": 64, "drop": 0.47, "lr": 0.0034},  # published
                {"h1": 128, "h2": 64, "drop": 0.1, "lr": 0.001},
                {"h1": 128, "h2": 64, "drop": 0.3, "lr": 0.003},
                {"h1": 256, "h2": 128, "drop": 0.3, "lr": 0.001},
                {"h1": 256, "h2": 128, "drop": 0.47, "lr": 0.0034},
                {"h1": 64, "h2": 32, "drop": 0.1, "lr": 0.01},
                {"h1": 512, "h2": 256, "drop": 0.3, "lr": 0.001},
                {"h1": 128, "h2": 64, "drop": 0.47, "lr": 0.01},
                {"h1": 256, "h2": 128, "drop": 0.1, "lr": 0.003},
                {"h1": 64, "h2": 32, "drop": 0.3, "lr": 0.0034},
            ],
            "build": lambda c: MLP2(len(contf), c["h1"], c["h2"], c["drop"]),
            "epochs": 5, "seq": False,
        },
        "lstm": {
            "configs": [
                {"hidden": 16, "lr": 0.01},  # published
                {"hidden": 32, "lr": 0.01},
                {"hidden": 64, "lr": 0.01},
                {"hidden": 128, "lr": 0.01},
                {"hidden": 16, "lr": 0.003},
                {"hidden": 32, "lr": 0.003},
                {"hidden": 64, "lr": 0.003},
                {"hidden": 128, "lr": 0.003},
                {"hidden": 64, "lr": 0.001},
                {"hidden": 128, "lr": 0.001},
            ],
            "build": lambda c: LSTM2(len(contf), c["hidden"]),
            "epochs": 3, "seq": True,
        },
        "transformer": {
            "configs": [
                {"d_model": 32, "layers": 2, "nhead": 4, "lr": 0.003},  # published
                {"d_model": 64, "layers": 2, "nhead": 4, "lr": 0.003},
                {"d_model": 64, "layers": 4, "nhead": 4, "lr": 0.001},
                {"d_model": 32, "layers": 4, "nhead": 4, "lr": 0.003},
                {"d_model": 64, "layers": 2, "nhead": 8, "lr": 0.001},
                {"d_model": 128, "layers": 2, "nhead": 4, "lr": 0.001},
                {"d_model": 32, "layers": 2, "nhead": 4, "lr": 0.01},
                {"d_model": 64, "layers": 4, "nhead": 8, "lr": 0.0003},
                {"d_model": 128, "layers": 4, "nhead": 8, "lr": 0.001},
                {"d_model": 32, "layers": 2, "nhead": 2, "lr": 0.001},
            ],
            "build": lambda c: TF2(len(contf), c["d_model"], c["nhead"], c["layers"]),
            "epochs": 5, "seq": False,
        },
    }

    sx22, sy22, tensors22 = make_tensors(tr22)
    sx23, sy23, tensors23 = make_tensors(tr23)

    for fam, spec in NET_FAMILIES.items():
        print(f"\n=== {fam} search (10 configs) ===")
        vals = []
        for i, cfg in enumerate(spec["configs"]):
            torch.manual_seed(42)
            model = spec["build"](cfg).to(device)
            train_net(model, tensors22, cfg["lr"], spec["epochs"], spec["seq"])
            maes = net_eval(model, sx22, sy22, val_h, spec["seq"])
            mean_mae = float(np.mean(list(maes.values())))
            vals.append(mean_mae)
            search_rows.append({"family": fam, "config_id": i, "published": i == 0,
                                "params": json.dumps(cfg), **{f"val_mae_{h}": maes[h] for h in EVAL_H},
                                "val_mae_mean": mean_mae})
            print(f"  cfg{i}{' (published)' if i == 0 else ''}: val MAE {mean_mae:.2f}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        best_i = int(np.argmin(vals))
        print(f"  selected: cfg{best_i}")
        for label, idx in [("selected", best_i), ("published", 0)]:
            if label == "published" and best_i == 0:
                continue
            cfg = spec["configs"][idx]
            torch.manual_seed(42)
            model = spec["build"](cfg).to(device)
            train_net(model, tensors23, cfg["lr"], spec["epochs"], spec["seq"])
            maes = net_eval(model, sx23, sy23, {h: te for h in EVAL_H}, spec["seq"])
            final_rows.append({"family": fam, "which": label, "config_id": idx,
                               "params": json.dumps(cfg), **{f"test_mae_{h}": maes[h] for h in EVAL_H},
                               "test_mae_mean": float(np.mean(list(maes.values())))})
            print(f"  {label} test MAE {np.mean(list(maes.values())):.2f}  {maes}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if best_i == 0:
            r = dict(final_rows[-1]); r["which"] = "published"; final_rows.append(r)

    pd.DataFrame(search_rows).to_csv(os.path.join(RESULTS_DIR, "reference_hparam_search.csv"), index=False)
    fr = pd.DataFrame(final_rows)
    fr["baseline_test_mae"] = baseline_test
    fr.to_csv(os.path.join(RESULTS_DIR, "reference_hparam_final.csv"), index=False)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. Saved reference_hparam_search.csv / reference_hparam_final.csv")


if __name__ == "__main__":
    main()
