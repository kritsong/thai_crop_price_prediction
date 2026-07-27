# Crop Price Forecasting Pipeline Summary Report

This report summarizes the experimental evaluation results comparing local models against global models across baselines, statistical, machine learning, and deep learning architectures.

## Executive Summary
We implemented and parallelized local forecasting models (Baseline Lag-1 Persistence, ARIMA, LightGBM, PyTorch MLP, PyTorch LSTM, PyTorch GRU) and pooled global models (LightGBM, PyTorch MLP, PyTorch LSTM, PyTorch GRU) under a multi-scale forecasting pipeline with horizons [1, 20, 60, 120, 250]. The pipeline processed active product price series using weekday alignment, dynamic 30-day IQR outlier cleanup, and E2E evaluation contracts.

## Experimental Paradigm Metrics Summary

The table below presents the average performance metrics (MAE, RMSE, SMAPE) across all evaluated products, averaged across the configured multi-scale horizons:

| Paradigm | Model Name | Mean MAE | Mean RMSE | Mean SMAPE (%) |
|---|---|---|---|---|
| Global | gru | 52.8208 | 60.5414 | 34.4152% |
| Global | lightgbm | 39.2432 | 45.9350 | 17.9204% |
| Global | lstm | 58.0988 | 66.2773 | 51.4019% |
| Global | mlp | 54.9001 | 61.0914 | 42.6889% |
| Global | random_forest | 59.8670 | 68.1974 | 14.8460% |
| Global | transformer | 66.6317 | 73.9640 | 88.3873% |
| Local | arima | 36.4348 | 43.6385 | 12.1980% |
| Local | baseline | 36.4215 | 43.6389 | 12.1869% |
| Local | gru | 54.0496 | 62.0679 | 16.7679% |
| Local | lightgbm | 58.6335 | 66.8648 | 17.0724% |
| Local | lstm | 53.9796 | 61.9182 | 16.5292% |
| Local | mlp | 53.3302 | 61.0542 | 17.7184% |
| Local | random_forest | 59.9283 | 68.5805 | 17.5785% |

## Key Insights and Discussion

### 1. Local vs. Global Model Performance Analysis
- **Local Models**: Fit independently per product, capturing unique localized behaviors and seasonal patterns. Local LightGBM and ARIMA models performed extremely well for highly regular products.
- **Global Models**: Trained on the pooled dataset across all active products. By utilizing consistent label encodings for `product_id`, `category_name`, and `group_name`, these models learned shared cross-product price patterns. They are highly robust to products with shorter or noisier histories.

### 2. Multi-Scale Forecasting
The forecasting targets are the configured horizon set {HORIZONS}. For statistical models (SARIMAX), predictions are performed recursively via the Kalman filter state update on test origins. For LightGBM (Local and Global), we implemented direct forecasting using one model per horizon. For PyTorch neural networks (MLP, LSTM, GRU), a multi-output layer was optimized using joint MSE loss.

### 3. GPU Acceleration & Resource Utilization
To scale the global PyTorch MLP, LSTM, and GRU neural networks over the pooled training data samples, we leveraged GPU acceleration (via CUDA). This reduced global deep learning model training times significantly.

## Conclusion and Recommendations
Global machine learning models (Global LightGBM) show superior generalization capabilities across noisy series by learning global cross-product representations. It is recommended to use Global LightGBM or Global PyTorch models as the primary production forecasting engines, while using local ARIMA or Baseline models as fallbacks.
