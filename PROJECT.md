# Project: Crop Price Forecasting Pipeline

This project implements a time series forecasting experiment pipeline to compare local models (trained per-product) against global models (a single shared model across products) using baseline, statistical, machine learning, and deep learning architectures. The raw workspace contains 729 product files; preprocessing filters out empty, static, and unusable series, and the final manuscript reports the 404-crop pure-price TFT experiments.

## Architecture

The pipeline follows a four-tier architecture design and processes the series sequentially or globally using available computing resources.

### Data Processing & Feature Engineering
- **Encoding & Parsing**: Data is read using `utf-8-sig` encoding, cleaning any BOM characters. Outliers are removed dynamically via rolling 30-day IQR bounds and filled.
- **Weekday Alignment**: The daily weekday price series is aligned with a continuous weekday skeleton (Monday-Friday business days). Holidays and reporting gaps are filled using forward-fill (`ffill()`) and backward-fill (`bfill()`).
- **Price-Only Features**: Features are derived exclusively from historical crop price data to prevent target leakage:
  - Lags: 1, 5, 7, 10, 20, 28, 40, 60, 70, 120, 125, and 250 business days.
  - Differences and returns: 1, 5, 20, 60, 120, and 250 business days.
  - Rolling statistics: 7-, 30-, 60-, and 120-day moving averages and standard deviations, plus 7-/30-day medians and 30-day range/CV features. All rolling price features are computed after shifting the target by 1 business day.
  - Flat-run indicators: days since last change and change counts over 20, 60, and 250 business days.
  - Calendar features: month, weekday, day-of-year, and cyclic encodings.
  - Categorical Metadata: Entity embeddings or label encoding for `product_id`, `group_name`, and `category_name` to support global models.

### Four-Tier Model Hierarchy
1. **Tier 1: Baseline (Lag-1 Persistence)**: Projects the previous business day's price forward.
2. **Tier 2: Statistical Models (ARIMA/SARIMAX)**: Local, parametric models trained independently for each product using price-only histories.
3. **Tier 3: Machine Learning Models (LightGBM/Random Forest)**: Tree-based ensemble regressors. Supports global modeling with label encoding and categorical splits.
4. **Tier 4: Deep Learning Models (PyTorch MLP/LSTM/GRU/Transformer and TFT)**: Neural network architectures utilizing GPU acceleration where available.

### Hardware & Resource Allocation
- **Local Models**: Parallelized over all available CPU threads using `joblib` multi-processing.
- **Global Models**: Evaluated efficiently using tabular trees (LightGBM) on CPU and sequential architectures (PyTorch) utilizing the RTX 4090 GPU via CUDA.

---

## Milestones

| # | Milestone Name | Scope | Dependencies | Status |
|---|----------------|-------|--------------|--------|
| 1 | M1: Environment & Prototypes | Verify execution environments (PyTorch CUDA, LightGBM, statsmodels), build feature engineering blocks, and create MLP prediction prototype. | None | DONE |
| 2 | M2: Baseline & Statistical Models | Implement Lag-1 persistence and local statistical models (ARIMA/SARIMAX or Exponential Smoothing) across all products. | M1 | DONE |
| 3 | M3: Machine Learning Models | Train local and global LightGBM and Random Forest models using price features and categorical embeddings. | M2 | DONE |
| 4 | M4: Deep Learning Models | Implement local and global MLP/LSTM/GRU/Transformer models and the TFT experiment flow. Leverage GPU acceleration (RTX 4090) for scaling. | M3 | DONE |
| 5 | M5: Pipeline Integration & Reporting | Consolidate predictions and evaluation metrics, generate boxplots of errors, and write the summary report. | M4 | DONE |
| 6 | M6: Verification & Auditing | Implement `verify_experiments.py` at the workspace root to check compliance and ensure clean execution. | M5 | DONE |

---

## Interface Contracts

### 1. Feature Engineering Contract
- Feature matrix must contain at least the following columns:
  - `lag_1`: $y_{t-1}$
  - `lag_7`: $y_{t-7}$
  - `lag_28`: $y_{t-28}$
  - `lag_70`: $y_{t-70}$
  - `roll_mean_7`: 7-day moving average of $y_{t-1 \dots t-7}$
  - `roll_std_7`: 7-day moving standard deviation of $y_{t-1 \dots t-7}$
  - `roll_mean_30`: 30-day moving average of $y_{t-1 \dots t-30}$
  - `roll_std_30`: 30-day moving standard deviation of $y_{t-1 \dots t-30}$
- Target column: `price_min` (or `price_max` depending on evaluation target, with next-day price as output).

### 2. Dataset Split Contract
- Temporal (out-of-time) splits:
  - **Training Set**: 2018-01-03 to 2023-12-29 (all weekday observations)
  - **Testing/Evaluation Set**: 2024-01-01 to 2025-12-30

### 3. Output CSV Specifications
All experiments must save predictions and evaluation metrics into a standardized format under `d:/new_crop_data/experiments_results/`:
- **Predictions CSV (`predictions.csv`)** columns:
  - `date` (YYYY-MM-DD target date)
  - `origin_date` (YYYY-MM-DD forecast origin date)
  - `horizon` (Integer business-day horizon)
  - `product_id` (String)
  - `actual_price` (Float)
  - `predicted_price` (Float)
  - `model_name` (String: e.g., "baseline", "sarimax", "lightgbm", "mlp")
  - `paradigm` (String: "local" or "global")
- **Metrics CSV (`metrics.csv`)** columns:
  - `product_id` (String)
  - `model_name` (String)
  - `paradigm` (String: "local" or "global")
  - `MAE` (Float)
  - `RMSE` (Float)
  - `SMAPE` (Float)
- **Horizon Metrics CSV (`metrics_by_horizon.csv`)** columns:
  - Same identifiers and metric columns as `metrics.csv`, plus `horizon`.

---

## Code Layout

- `d:/new_crop_data/`:
  - `src/`
    - `prototype.py` - Milestone 1 execution prototype
    - `data/`
      - `loader.py` - Data loading and alignment helper
    - `features/`
      - `generator.py` - Feature engineering pipelines
    - `models/`
      - `train.py` - Unified training loop (local & global paradigms)
  - `experiments_results/` - Output directory
    - `predictions.csv` - Unified forecast records
    - `metrics.csv` - Evaluation results by product and model
    - `metrics_by_horizon.csv` - Evaluation results by product, model, and horizon
    - `summary_report.md` - Final project report
  - `verify_experiments.py` - Verification suite for milestone outputs
  - `PROJECT.md` - Project architecture and contracts (this file)
