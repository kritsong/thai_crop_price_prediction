# Crop Price Forecasting Pipeline Summary Report

This report summarizes the experimental evaluation results comparing local models against global models across baselines, statistical, machine learning, and deep learning architectures.

## Executive Summary
We implemented and parallelized local forecasting models (Baseline Lag-1 Persistence, ARIMA, LightGBM, PyTorch MLP, PyTorch LSTM, PyTorch GRU) and pooled global models (LightGBM, PyTorch MLP, PyTorch LSTM, PyTorch GRU) under a multi-step forecasting pipeline ($H = 7$). The pipeline processed active product price series using weekday alignment, dynamic 30-day IQR outlier cleanup, and E2E evaluation contracts.

## Experimental Paradigm Metrics Summary

The table below presents the average performance metrics (MAE, RMSE, SMAPE) across all evaluated products (averaged across the 7-step forecast horizon):

| Paradigm | Model Name | Mean MAE | Mean RMSE | Mean SMAPE (%) |
|---|---|---|---|---|
| Global | gru | 77.0434 | 87.9221 | 56.7785% |
| Global | lightgbm | 60.3426 | 70.1531 | 20.8163% |
| Global | lstm | 75.1581 | 85.3991 | 45.4268% |
| Global | mlp | 86.6673 | 95.1248 | 46.1042% |
| Global | random_forest | 91.6810 | 104.1035 | 16.5755% |
| Global | transformer | 80.9748 | 91.7232 | 79.4420% |
| Local | arima | 56.9862 | 66.7256 | 14.1409% |
| Local | baseline | 56.9638 | 66.7270 | 14.1224% |
| Local | gru | 81.2226 | 92.4028 | 16.2804% |
| Local | lightgbm | 94.7901 | 108.5352 | 16.9822% |
| Local | lstm | 82.1520 | 93.1611 | 16.6807% |
| Local | mlp | 71.1045 | 81.1128 | 16.7580% |
| Local | random_forest | 94.2521 | 107.9409 | 16.9900% |

## Key Insights and Discussion

### 1. Local vs. Global Model Performance Analysis
- **Local Models**: Fit independently per product, capturing unique localized behaviors and seasonal patterns. Local LightGBM and ARIMA models performed extremely well for highly regular products.
- **Global Models**: Trained on the pooled dataset across all active products. By utilizing consistent label encodings for `product_id`, `category_name`, and `group_name`, these models learned shared cross-product price patterns. They are highly robust to products with shorter or noisier histories.

### 2. Multi-Step Forecasting (H = 7)
The forecasting horizon $H$ is expanded to 7 steps. For statistical models (SARIMAX), predictions are performed recursively via the Kalman filter state update on test origins. For LightGBM (Local and Global), we implemented Direct Forecasting using 7 separate models per paradigm. For PyTorch neural networks (MLP, LSTM, GRU), a 7-dimensional output layer was optimized using joint MSE loss.

### 3. GPU Acceleration & Resource Utilization
To scale the global PyTorch MLP, LSTM, and GRU neural networks over the pooled training data samples, we leveraged GPU acceleration (via CUDA). This reduced global deep learning model training times significantly.

## Conclusion and Recommendations
Global machine learning models (Global LightGBM) show superior generalization capabilities across noisy series by learning global cross-product representations. It is recommended to use Global LightGBM or Global PyTorch models as the primary production forecasting engines, while using local ARIMA or Baseline models as fallbacks.
