# Horizon-Weighted Temporal Fusion Transformers for Thai Agricultural Commodity Price Forecasting

Code and reproduction scripts for the paper *"Beyond the Random Walk:
Horizon-Weighted Temporal Fusion Transformers for Thai Agricultural Commodity
Price Forecasting."*

The paper asks why pooled deep models routinely lose to a naive
last-observed-price rule on low signal-to-noise agricultural price series, and
introduces a **Horizon-Weighted Quantile Loss**, `w(h) = 1/h^gamma`, that
resolves the multi-scale gradient conflict responsible. Evaluation covers 404
Thai crop products from 2018 to 2025 under a matched-window protocol, against
persistence, seasonal-naive, drift, ARIMA, LightGBM, MLP, LSTM, Transformer and
a zero-shot time-series foundation model.

---

## 1. Setup

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

A CUDA GPU is needed to retrain the TFT or to run the foundation-model baseline.
Every analysis step below runs on CPU from saved artefacts.

### Where the data and artefacts live

`paths.py` resolves the two directories that sit outside version control, in
this order: an environment variable, then a repository-local directory, then a
sibling directory beside the repository.

| Variable | Contents | Repository-local default |
| :--- | :--- | :--- |
| `CROP_DATA_DIR` | raw `*.json` price files, one per product | `./data/historical_data_2018` |
| `CROP_EXPERIMENTS_DIR` | checkpoints and saved per-window predictions | `./experiments_results` |

```bash
python paths.py          # prints the resolved paths and whether they exist
```

Large artefacts are excluded from git. Rebuilding the selected-model tables and
figures without retraining requires its checkpoint
(`tft_hw_quantile_gamma_4_5.ckpt`, 1.1 MB) and saved per-window predictions
(`tft_hw_quantile_gamma_4_5_predictions.npz`, 9.7 MB). Rebuilding the complete
17-point sweep additionally requires the corresponding per-gamma metric and
prediction files. Version-controlled aggregate CSV files are provided so the
published tables and sweep figure remain auditable without those large files.

---

## 2. Reproducing the paper

### Tables

| Table | Content | Command |
| :--- | :--- | :--- |
| 1, 2 | product inventory, stationarity | `python run_eda.py` |
| 3, 4 | data partition, model configuration | descriptive, no script |
| 5 | reference suite | `python -m src.models.train`, then `python run_extra_baselines.py` and `python run_chronos_zeroshot.py` |
| 6 | decay-exponent sweep | `python run_gamma_sweep.py` |
| 7 | selected model vs baseline, with significance tests | `python -m src.models.train_tft --gamma 4.5`, then `python build_publication_metrics.py` |
| 8, 9 | temporal stability, calibrated intervals | `python build_publication_metrics.py` |

### Figures

| Figure | Content | Command |
| :--- | :--- | :--- |
| 1-4 | ACF/PACF, price trends, stationarity, correlation | `python run_eda.py` |
| 5 | loss weighting decay curves | `python generate_loss_curve_figure.py` |
| 6 | MAE against decay exponent | `python generate_gamma_sweep_figure.py` |
| 7 | qualitative forecasts, deliberately extreme cases | `python generate_qualitative_figure.py` |
| 8 | qualitative forecasts, one group-median case per commodity group | `python generate_typical_qualitative_figure.py` |
| 9-12 | attention and variable selection | `python generate_interpretability_figures.py` |

### Supplementary material

```bash
python run_reference_search.py     # equal-budget hyperparameter search for every reference class
python build_supplement.py         # writes paper/supplementary.tex
```

### Building the documents

```bash
python build_latex.py        && (cd paper && pdflatex main.tex && pdflatex main.tex)
python build_supplement.py   && (cd paper && pdflatex supplementary.tex)
python build_cover_letter.py && (cd paper && pdflatex cover_letter.tex)
```

### Verification

```bash
python verify_experiments.py   # 88 unit and end-to-end tests over the pipeline
python verify_keywords.py      # manuscript integrity: figure paths and embedded-image count
python verify_publication.py   # journal limits, placeholders, artifacts and PDFs
```

After all automated checks pass, complete the author-only confirmations in
`SUBMISSION_CHECKLIST.md`. Those declarations require the authors' direct
approval and cannot be inferred from the repository.

---

## 3. Repository layout

```
paths.py                       central path resolution, used by every script
src/data/                      loading, cleaning, business-day alignment
src/features/generator.py      leakage-safe feature construction
src/models/loss.py             HorizonWeightedQuantileLoss
src/models/train_tft.py        TFT training, evaluation, interval calibration
src/models/train.py            reference models (ARIMA, LightGBM, MLP, LSTM, GRU, Transformer)
results/                       aggregated metrics consumed by the manuscript
paper/                         manuscript source, figures and built PDFs
```

Two implementation details worth knowing before modifying anything:

- **`HorizonWeightedQuantileLoss` subclasses `QuantileLoss`, not
  `MultiHorizonMetric`.** `pytorch_forecasting` deduces `output_size` from an
  `isinstance` check against `QuantileLoss`. Subclassing the wrong base silently
  produces `output_size=1` and the model fails at runtime.
- **The target normaliser is fitted once on the training window and never
  refreshed.** This is a real limitation, quantified rather than hidden, in
  Section 4.8 of the paper.

---

## 4. Data provenance and licence

Raw prices come from the Ministry of Commerce (Thailand), Department of Internal
Trade, via the MOC Open Data portal, "Agricultural Product Price" dataset.
Check the portal's own terms before redistributing the raw files; if the terms
are unclear, run the pipeline against a fresh download rather than a mirrored
copy.

The code here is released under the licence in `LICENSE`. The underlying price
data remains the property of its original publisher and is not covered by that
licence.

---

## 5. Citation

Citation details will be added on publication. Until then, contact Kritaphat
Songsri-in at `kritaphat_son@nstru.ac.th`.
