# Thai crop price forecasting with knowledge-guided mean reversion

Code for our paper "Beyond the Random Walk: Knowledge-Guided Mean Reversion for
Multi-Horizon Thai Agricultural Commodity Price Forecasting".

Daily prices of 404 Thai crop products (2018 to 2025, Ministry of Commerce)
behave close to a random walk, and at horizons of one month to one year almost
nothing beats the naive "the future price is today's price" forecast for the
typical product. We benchmark 23 references against it, from ARIMA and LightGBM
to a Temporal Fusion Transformer (TFT) and zero-shot and fine-tuned
Chronos and TimesFM foundation models, and show that averaging errors in Thai
baht hides this failure, because wholesale rice is 11% of the products but 73%
of the persistence error.

The proposed forecaster is small and interpretable. It moves the last observed
price part of the way toward a recent (750-day) and a long-run (expanding) price
level,

```
y_hat = y + a * (mu_short - y) + b * (mu_long - y),   a, b >= 0,  a + b <= 1
```

with shares fitted per commodity group, horizon, and deviation state by least
absolute deviations. The design is chosen by rolling-origin validation over
2021 to 2023 and tested on 2024 to 2025. The implementation is
`src/models/mean_reversion.py`.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The mean-reversion forecaster runs on CPU in minutes. Retraining the TFT or
running the foundation models needs a GPU.

### Data and prediction files

`paths.py` looks for the data and the large experiment files in this order: an
environment variable, a folder inside the repo, then a folder next to it.

| Variable | What it holds | Default |
| :--- | :--- | :--- |
| `CROP_DATA_DIR` | the raw `*.json` price files | `./data/historical_data_2018` |
| `CROP_EXPERIMENTS_DIR` | checkpoints and per-window prediction files | `./experiments_results` |

Run `python paths.py` to see what it resolved.

The raw price files are included under `data/historical_data_2018/`. The
checkpoints and per-window prediction files (`*.npz`) are not, because of their
size. The evaluation scripts read the matched test windows from the saved
TFT predictions (`tft_hw_quantile_gamma_4_5_predictions.npz`),
so regenerate that file with `python -m src.models.train_tft --gamma 4.5` or ask
the corresponding author for a copy. Every table in the paper can be checked
without these files from the CSV and JSON files in `results/`.

## Reproducing the paper

The proposed forecaster and its evaluation:

```bash
python experiments/run_mean_reversion.py      # first single-year selection (Table S4)
python experiments/run_mean_reversion_v2.py   # second single-year selection (Table S5)
python experiments/run_mean_reversion_cv.py   # rolling-origin selection, the proposed model (Table 2, Table S3)
python experiments/run_mr_evaluation.py       # Tables 3 to 7 and the mr_*.csv files
python figures/generate_mr_figures.py         # Figures 1, 2, S5 and S6
```

References:

```bash
python experiments/run_eda.py                 # dataset statistics and the exploratory figures
python -m src.models.train                    # classical and machine learning references
python experiments/run_extra_baselines.py     # seasonal-naive and drift
python experiments/run_reference_search.py    # ten-configuration searches (Supplementary Section S7)
python experiments/run_gamma_sweep.py         # TFT loss-exponent sweep (Supplementary Section S8)
python -m src.models.train_tft --gamma 4.5    # TFT, loss weighted by horizon (gamma 4.5)
python experiments/run_tft_variant.py ...     # redesigned TFT variants (Table 8), see the script's help
python experiments/eval_variants.py           # scores the TFT variants
python experiments/run_chronos_zeroshot.py    # Chronos-Bolt, zero-shot
python experiments/run_chronos2.py            # Chronos-2, zero-shot and fine-tuned
python experiments/run_timesfm_zeroshot.py --version 2.5
python experiments/run_timesfm_zeroshot.py --version 3.0
python experiments/run_timesfm_finetune.py    # TimesFM 2.5 with LoRA
```

Tests:

```bash
python tests/verify_experiments.py
```

## Layout

```
paths.py          where the data and prediction files live
src/data/         loading, cleaning, business-day alignment
src/features/     feature construction for the learned references
src/models/       mean reversion, the TFT and its variants, the reference models
experiments/      the runs that produce the numbers
figures/          the scripts that draw each figure
tests/            the pipeline test suite
results/          the numbers behind every table
```

Scripts run either way, `python experiments/run_eda.py` or
`python -m experiments.run_eda`, from the repository root.

The mean-reversion forecaster has no random initialization, so its results are
deterministic. `run_mr_evaluation.py` asserts that refitting reproduces the saved
forecasts exactly and that the seasonal-naive, drift, and unweighted TFT
references reproduce their published aggregates.

## Data licence

Prices are from the Ministry of Commerce (Thailand), Department of Internal
Trade, via the MOC Open Data portal ("Agricultural Product Price"). The copy
under `data/` is included so this work can be reproduced and is not offered
under the repository's licence. The code is MIT licensed (see `LICENSE`). The
price data belongs to the Ministry, is not covered by that licence, and remains
subject to the Ministry's terms. Check with the Ministry before reusing it for
anything beyond reproducing this study, and treat the portal as the
authoritative source.

## Contact

Kritaphat Songsri-in, `kritaphat_son@nstru.ac.th`. Citation details will follow
once the paper is published.
