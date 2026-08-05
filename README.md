# Thai crop price forecasting with a horizon-weighted TFT

Code for our paper on forecasting Thai agricultural commodity prices with a
Temporal Fusion Transformer.

The starting point was a frustrating result: on these price series, pooled deep
models kept losing to the naive "tomorrow's price is today's price" rule. Our fix
is a horizon-weighted quantile loss, `w(h) = 1/h^gamma`, which stops the
long-horizon steps from dominating training. We sweep gamma from 0 to 8 and pick
the operating point on a held-out validation year.

Data is 404 crop products, 2018 to 2025, from the Ministry of Commerce. We
compare against persistence, seasonal-naive, drift, ARIMA, LightGBM, MLP, LSTM,
a plain Transformer, and a zero-shot foundation model.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

You need a GPU to retrain the TFT or run the Chronos baseline. Everything else
runs fine on CPU from the saved result files.

### Data and checkpoint locations

The raw price files and the model checkpoints are not in this repo. `paths.py`
looks for them in this order: an environment variable, a folder inside the repo,
then a folder next to it.

| Variable | What it holds | Default |
| :--- | :--- | :--- |
| `CROP_DATA_DIR` | the raw `*.json` price files | `./data/historical_data_2018` |
| `CROP_EXPERIMENTS_DIR` | checkpoints and saved predictions | `./experiments_results` |

Run `python paths.py` to see what it resolved and whether those folders exist.

The raw data comes from the MOC Open Data portal ("Agricultural Product Price").
We do not mirror it here, so download it yourself and point `CROP_DATA_DIR` at it.

## Running things

Experiments:

```bash
python experiments/run_eda.py               # dataset stats and the EDA figures
python -m src.models.train                  # the reference models
python experiments/run_extra_baselines.py   # seasonal-naive and drift
python experiments/run_chronos_zeroshot.py  # zero-shot foundation model (needs GPU)
python experiments/run_gamma_sweep.py       # the gamma sweep
python experiments/run_reference_search.py  # hyperparameter search for the references
python experiments/run_dm_tests.py          # significance tests
```

Training the main model:

```bash
python -m src.models.train_tft --gamma 4.5
python experiments/build_publication_metrics.py
```

Figures:

```bash
python figures/generate_loss_curve_figure.py
python figures/generate_gamma_sweep_figure.py
python figures/generate_qualitative_figure.py
python figures/generate_typical_qualitative_figure.py
python figures/generate_interpretability_figures.py
```

Tests:

```bash
python tests/verify_experiments.py
```

## Layout

```
paths.py          where the data and checkpoints live
src/data/         loading, cleaning, business-day alignment
src/features/     feature construction
src/models/       the horizon-weighted loss, TFT training, reference models
experiments/      the runs that produce the numbers
figures/          the scripts that draw each figure
tests/            the pipeline test suite
results/          the numbers behind the paper's tables
```

Scripts work either way, `python experiments/run_eda.py` or
`python -m experiments.run_eda`, as long as you are at the repository root.

Two things that will bite you if you change the code:

`HorizonWeightedQuantileLoss` subclasses `QuantileLoss`, not `MultiHorizonMetric`.
`pytorch_forecasting` uses an `isinstance` check to work out `output_size`, so
picking the wrong base class silently gives you `output_size=1` and a crash later.

The target normaliser is fitted once on the training window and never refreshed.
That is a real weakness and we discuss it in the paper rather than paper over it.

## Data licence

Prices are from the Ministry of Commerce (Thailand), Department of Internal
Trade. Check their terms before redistributing the raw files. The code here is
MIT licensed (see `LICENSE`); the price data is not ours and is not covered by it.

## Contact

Kritaphat Songsri-in, `kritaphat_son@nstru.ac.th`. Citation details will follow
once the paper is out.
