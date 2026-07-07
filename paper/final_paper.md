# Decoupling Multi-Scale Agricultural Price Dynamics: A Pooled Time-Series Fusion Transformer Approach

**Authors:** Academic Forecasting Syndicate  
**Affiliation:** Department of Quantitative Economics and Machine Learning Research  
**Date:** June 21, 2026  

---

## Abstract

Predicting agricultural commodity prices is a challenging task due to non-linear dynamics, market inefficiencies, and sudden price movements. In developing markets like Thailand, crop price volatility poses risks to farmers, traders, and policymakers. Historically, traditional forecasting methods and standard machine learning architectures have struggled to outperform simple local baselines, often leading researchers to conclude that these markets behave as random walks. This study challenges this assumption by evaluating a global pooled Temporal Fusion Transformer (TFT) architecture over multi-scale horizons ($t+20$, $t+60$, $t+120$, and $t+250$ business days) across hundreds of Thai crop products.

We identify a fundamental challenge in multi-horizon sequence prediction: multi-scale gradient conflict, where the large error magnitudes of far-future steps dominate backpropagation, causing the network to ignore crucial short-term price transmission dynamics. To resolve this gradient scale imbalance, we propose a **Horizon-Weighted Loss** optimization paradigm that dynamically scales step-dependent gradients. We systematically parameterize and evaluate the loss decay weighting function using a 1-parameter power-law family $w(h) = 1/h^\gamma$ across seven configurations spanning $\gamma \in \{0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0\}$. Our empirical results demonstrate that the proposed weighted architectures successfully break the random walk baseline out-of-sample (2024–2025). The champion Inverse-Square model ($\gamma = 2.0$) achieves the lowest overall average MAE of **40.15 THB** (outperforming the overall Lag-1 persistence baseline of **42.23 THB** and yielding a **5.97% error reduction** over standard unweighted TFT). In the short term ($t+20$ business days), due to high daily price autocorrelation, the local Lag-1 persistence baseline remains the top predictor with an MAE of **15.76 THB**; however, our horizon-weighted loss successfully bridges this gap, reducing near-term error by **32.2%** (dropping MAE from **29.07 THB** for standard unweighted TFT to **19.72 THB**). In the long term ($t+250$ business days), where drift variance accumulates, the global models dominate, with the Linear model ($\gamma = 1.0$) achieving the best one-year MAE of **54.97 THB** (a **24.8% error reduction** compared to the Lag-1 persistence baseline's **73.08 THB**). The TFT leverages Variable Selection Networks (VSN) and Interpretable Multi-Head Self-Attention to capture complex dependencies, providing full model interpretability.

## 1. Introduction

Agricultural commodity markets in developing nations, particularly Thailand, are characterized by high price volatility. As a leading exporter of rice, cassava, and other crops, Thailand's economic stability is tied to agricultural pricing. However, these prices are subject to complex domestic supply shocks, international trade policies, and market inefficiencies. Reliable price forecasting over multiple horizons is essential for risk mitigation, farm-level decision-making, and policy formulation.

Under the efficient market hypothesis, agricultural asset prices are theorized to follow random walks, rendering historical price-only structures uninformative for future horizons. Empirically, this is reflected in the difficulty that high-capacity models encounter when trying to outperform naive local baselines out-of-sample. This phenomenon, which we term 'architectural saturation,' arises because complex neural architectures (e.g., standard sequence-to-sequence Transformers, deep MLPs, and recurrent networks) possess high parameter density and capacity, allowing them to fit arbitrary functions. However, agricultural price series are characterized by low signal-to-noise ratios, non-stationarity, and frequent structural breaks. When trained on such series, high-capacity models tend to memorize transient noise and spurious historical correlations within the training sample rather than extracting persistent, structural patterns. Consequently, when evaluated on out-of-sample hold-out periods (such as 2024–2025), these over-parameterized models suffer from generalization collapse, performing significantly worse than a simple Lag-1 persistence model, which remains robust due to its zero-parameter simplicity.

To overcome the limitations of localized statistical modeling and mitigate the risk of architectural saturation, we leverage cross-product representation learning using a global Temporal Fusion Transformer (TFT) architecture. Traditional local models are trained independently on each of the 404 crops, meaning they cannot share statistical strength and are highly vulnerable to localized data scarcity or reporting anomalies. In contrast, our global modeling framework pools the price histories of all 404 crops into a single training dataset, training a single set of shared model parameters. To prevent the model from conflating different crop dynamics, we feed static variables (such as product, category, and group IDs) into a static covariate encoder, mapping them to low-dimensional embedding spaces. This representation learning paradigm allows the TFT to capture universal, shared features of agricultural price dynamics—such as macro-level price transmission, shared volatility regimes, and general seasonal cycles—while simultaneously conditioning its forecasts on entity-specific static embeddings to preserve the unique price-level characteristics of each individual crop.

The primary contributions of this paper are:
1. We establish a multi-scale forecasting skeleton targeting four distinct future horizons: $t+20$ (one business month), $t+60$ (three business months), $t+120$ (six business months), and $t+250$ (one business year) to capture agricultural price dynamics.
2. We identify and formulate the multi-scale gradient scale imbalance in joint sequence prediction, showing how long-term loss errors dominate sequence learning at the expense of near-term persistence structures.
3. We present a Horizon-Weighted Loss optimization framework parameterized by a 1-parameter power-law family $w(h) = 1/h^\gamma$ to balance gradients across lookahead horizons without losing the parameter-sharing advantages of a single shared network.
4. We show that the proposed weighted TFT architectures successfully break the random walk ceiling, achieving significant error reductions over the unweighted TFT and naive persistences, and we decode their internal attention spans to provide economic interpretability.

The remainder of this paper is structured as follows. Section 2 reviews the literature on agricultural price forecasting. Section 3 details the dataset and exploratory data analysis. Section 4 describes the methodology and forecasting models. Section 5 presents the quantitative results, probabilistic forecasting evaluations, and ablation studies. Section 6 discusses the implications and interpretability. Section 7 concludes with future work.

## 2. Literature Review

Historically, agricultural price forecasting has been dominated by localized, parametric statistical models. The Autoregressive Integrated Moving Average (ARIMA) framework, popularized by Box and Jenkins (1970), serves as a classical benchmark. These local models are fitted independently to each individual commodity series under the assumption of linear, stationary data-generating processes. While ARIMA and its seasonal variants (SARIMA) provide robust predictions in stable economic regimes, they fail to capture the high-frequency non-linear dynamics, volatility clustering, and sudden structural breaks characteristic of developing agricultural markets (Gooijer & Hyndman, 2006). Furthermore, because these statistical methods are localized, they cannot share parameters across different commodities. This severely limits their predictive capacity when individual series are short, incomplete, or characterized by low signal-to-noise ratios.

To capture non-linear relationships, researchers transitioned to machine learning (ML) paradigms, such as Support Vector Regression (SVR), Random Forests, and Gradient Boosting Decision Trees (GBDT), including XGBoost and LightGBM. GBDTs have shown strong empirical performance on tabular forecasting tasks (Ke et al., 2017) due to their ability to construct non-linear decision boundaries. However, standard ML models treat sequence prediction as a static regression problem, requiring extensive manual feature engineering of rolling windows and historical lags. In a global setting—where a single model is trained across a diverse cross-section of products—traditional GBDTs struggle to scale efficiently. They often experience severe overfitting to localized noise rather than learning generalizable sequential relationships, a precursor to architectural saturation out-of-sample.

The advent of deep learning (DL) provided a framework for end-to-end representation learning. Recurrent Neural Networks (RNNs), particularly Long Short-Term Memory (LSTM) networks (Hochreiter & Schmidhuber, 1997) and Gated Recurrent Units (GRUs), introduced gated cell states to capture sequential dependencies. While RNNs address sequence learning, they suffer from vanishing gradients over long lookback windows and process inputs sequentially, which limits parallelization during training. The self-attention mechanism of the Transformer (Vaswani et al., 2017) resolved these bottlenecks, enabling the modeling of long-range interactions. Nevertheless, standard Transformers are highly over-parameterized and lack specialized mechanisms to handle the distinct properties of time-series data, such as static metadata and time-varying covariates, often resulting in architectural saturation where the model overfits to historical noise and fails to generalize out-of-sample.

To bridge this gap, the Temporal Fusion Transformer (TFT) was developed (Lim et al., 2021). The TFT combines self-attention with Gated Residual Networks (GRN) and Variable Selection Networks (VSN). The VSN acts as an active information filter, selecting relevant features at each step, while the GRN provides adaptive model capacity by bypassing unnecessary layers. This architectural flexibility is critical for agricultural price forecasting, where the signal-to-noise ratio is low and overfitting is common. Furthermore, the TFT introduces interpretable self-attention, allowing researchers to inspect the exact historical steps the model prioritized when generating forecasts. Similar architectures, such as PatchTST (Nie et al., 2023) and iTransformer (Liu et al., 2024), have focused on channel-independence or patching to capture local sequences. However, they lack the native integration of static metadata (e.g., crop class, market category) and dynamic covariates in a unified framework, which is crucial for transfer learning across heterogeneous agricultural markets. Similarly, large pre-trained foundation models like Chronos (Ansari et al., 2024) leverage zero-shot forecasting, but they can suffer from sequence hallucinations over long horizons when decoupled from localized, entity-specific economic context.

## 3. Data & EDA

Our dataset comprises daily price recordings of agricultural products in Thailand. To ensure the models focus on economically active commodities rather than static or inactive crop listings, we apply a volatility filtering mechanism. Static, "step-function" products that show zero price movement over long periods due to price controls or illiquidity are filtered out.

We apply a **1% Volatility Threshold**, which requires a crop product to experience a non-zero price change on at least 1% of its active trading days. This filter retains **404 crops** for our experiments.

### 3.1. Data Cleaning and Preprocessing
Raw agricultural price series often contain reporting anomalies and data gaps. An inspection of the dataset revealed that **4.60%** of the raw daily price observations contained drops to exactly zero (with **12.89%** of the series being entirely empty or all-zero and subsequently excluded). These zero values do not represent real market crashes but are artifacts of missing reports or database logging errors. 

To clean the data:
1. All zero prices are cast to `NaN` (Not a Number) values.
2. We apply a Last-Observation-Carried-Forward (LOCF) forward-fill to impute the missing values, ensuring that the last known valid market price is propagated.
3. The dataset is aligned to a standard Monday-Friday weekly business calendar skeleton to remove weekend gaps.

### 3.2. Product Classification and Quality Profiling
The dataset covers a wide range of agricultural products in Thai markets. To profile the completeness across different commodity groups, Table 1 details the active and empty/all-zero series count for each group.

#### Table 1: Product Counts and Status by Group
| Group Name | Active Products | All-Zero/Empty Products | Total Products |
|---|---|---|---|
| Meat & Poultry | 98 | 14 | 112 |
| Aquatic & Marine | 39 | 6 | 45 |
| Fresh Vegetables | 89 | 0 | 89 |
| Organic Vegetables & Fruits | 44 | 0 | 44 |
| Fruits | 64 | 8 | 72 |
| Food Crops | 82 | 11 | 93 |
| Oil Crops & Oils | 95 | 41 | 136 |
| Wholesale Rice & Bags | 54 | 0 | 54 |
| Wholesale Rice to Retailers | 10 | 0 | 10 |
| Retail Rice | 9 | 1 | 10 |
| Field Crops | 40 | 4 | 44 |
| Feed & Raw Feed Materials | 11 | 4 | 15 |
| Miscellaneous / Unknown | 0 | 5 | 5 |
| **Total** | **635** | **94** | **729** |

### 3.3. Time Series Properties and Stationarity Analysis
Unit root properties vary significantly across commodity classes. Table 2 presents the percentage of stationary minimum and maximum price series across retail and wholesale categories, demonstrating that wholesale prices generally exhibit lower stationarity rates than retail prices.

#### Table 2: Stationarity Profiles by Market Category
| Category | Active Series | Stationary Minimum Price (%) | Stationary Maximum Price (%) |
|---|---|---|---|
| Retail | 269 | 36.8% | 34.6% |
| Wholesale | 366 | 28.1% | 26.0% |

### 3.4. Exploratory Data Analysis

We perform a thorough exploratory analysis to understand the statistical properties of the price series. 

First, we analyze the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) to identify the lag structure and check for random walk characteristics.

![Autocorrelation and Partial Autocorrelation Functions](images/acf_pacf.png)  
*Figure 1: Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots for representative crop price series.*

Next, we inspect the long-term price trends across different product categories to identify macro-level movements and historical volatility regimes.

![Price Trend and Volatility Analysis](images/price_trend.png)  
*Figure 2: Historical price trajectories showing volatility regimes across major crop categories.*

To evaluate stationarity, we conduct Augmented Dickey-Fuller (ADF) unit-root tests on the minimum and maximum price series. We aggregate the share of stationary series by product classification to compare differences across market categories.

![Stationarity Profiles by Group](images/stationarity_by_group.png)  
*Figure 3: ADF stationarity rates for active minimum and maximum price series by product group. Thai group names are translated to concise English labels for publication readability; n denotes the number of active series.*

The stationarity analysis reveals that a large percentage of the raw price series are non-stationary in levels, requiring robust models that can handle trend-drift and structural shifts without producing spurious regressions.

Finally, we analyze the cross-correlations and co-movements between different crops to understand market integration and price transmission. The correlation heatmap illustrates the pairwise Pearson correlation coefficients calculated over the synchronized price trajectories.

![Cross-Crop Price Correlation Heatmap](images/correlation_heatmap.png)  
*Figure 4: Pairwise Pearson correlation heatmap showing cross-crop price-price correlations.*

The heatmap reveals strong blocks of cross-correlations among related crop groups (e.g., different varieties of rice or cassava products). This high degree of market integration indicates that price movements in one crop are co-dependent on prices of other crops, validating the use of global, pooled models that learn shared representations across all commodities.

## 4. Methodology

### 4.1. Forecasting Horizon and Evaluation Metrics
We establish a multi-scale forecasting horizon aligned with agricultural decision cycles. Given the input sequence, models forecast at specific future steps:
*   $t+20$: One business month / 20 business days.
*   $t+60$: Three business months / 60 business days.
*   $t+120$: Six business months / 120 business days.
*   $t+250$: One business year / 250 business days.

Performance is measured using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Symmetric Mean Absolute Percentage Error (SMAPE), calculated as:

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$

$$\text{SMAPE} = \frac{100\%}{N} \sum_{i=1}^{N} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}$$

where $y_i$ represents the actual price and $\hat{y}_i$ represents the predicted price.

### 4.2. Baseline and Global Models

To evaluate the predictive capacity of the Time-Series Fusion Transformer (TFT), we compare it against a diverse suite of localized statistical benchmarks and global machine learning and deep learning models:

1. **Lag-1 Persistence Baseline:**
   The Lag-1 persistence baseline represents the naive forecast for a random walk. It assumes that the future price at any future horizon $t+h$ is identical to the most recently observed price at time $t$:
   $$\hat{y}_{t+h} = y_t$$
   Despite its zero-parameter simplicity, the persistence model is a highly competitive benchmark in financial and commodity forecasting due to the high day-to-day autocorrelation in market prices.

2. **Local ARIMA:**
   The Autoregressive Integrated Moving Average (ARIMA) model, popularized by Box and Jenkins (1970), is fitted individually to each of the 404 crop price series. An $\text{ARIMA}(p, d, q)$ model is represented as:
   $$\Delta^d y_t = c + \sum_{i=1}^p \phi_i \Delta^d y_{t-i} + \sum_{j=1}^q \theta_j \epsilon_{t-j} + \epsilon_t$$
   where $\Delta^d$ is the difference operator of order $d$, $\phi_i$ are autoregressive coefficients representing sequential dependence, $\theta_j$ are moving average coefficients capturing short-term shocks, and $\epsilon_t$ is white noise. While ARIMA is a powerful local predictor, its linear assumptions and inability to share parameters across different series limit its capacity to model non-linear price transmission and market-wide volatility regimes.

3. **Global MLP:**
   The Global Multi-Layer Perceptron (MLP) is a feedforward neural network trained across the pooled dataset of all crops. Given a flattened vector of historical price lags $\mathbf{x}_{t-k:t} \in \mathbb{R}^k$, the MLP maps the input to the multi-scale prediction horizons through $L$ fully-connected hidden layers:
   $$\mathbf{h}^{(l)} = g\left(\mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}\right), \quad l=1,\dots,L$$
   where $\mathbf{h}^{(0)} = \mathbf{x}_{t-k:t}$, $\mathbf{W}^{(l)}$ and $\mathbf{b}^{(l)}$ are weights and biases, and $g(\cdot)$ is the Rectified Linear Unit (ReLU) activation function. The final layer outputs the point forecasts.

4. **Global LightGBM:**
   LightGBM (Ke et al., 2017) is an optimized Gradient Boosting Decision Tree (GBDT) framework. In our multi-scale setup, we train 4 independent GBDT models (one for each horizon $h \in \{20, 60, 120, 250\}$). The prediction is an ensemble of $M$ decision trees:
   $$\hat{y}_{t+h} = \sum_{m=1}^M f_m(\mathbf{x}_{t-k:t})$$
   where each leaf-wise split $f_m$ is trained to minimize the mean absolute error by fitting the residual gradients of the preceding trees.

5. **Global LSTM and GRU:**
   These recurrent architectures process the sequential historical lags using gated cell structures. They are trained globally across all commodities to capture shared sequential dynamics. The Gated Recurrent Unit (GRU) simplifies this structure by merging the cell state and hidden state, governed by:
   $$\mathbf{z}_t = \sigma(\mathbf{W}_z [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_z)$$
   $$\mathbf{r}_t = \sigma(\mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_r)$$
   $$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_h)$$
   $$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$
   where $\mathbf{z}_t$ is the update gate and $\mathbf{r}_t$ is the reset gate. Unlike the TFT, LSTMs and GRUs lack direct self-attention layers and variable selection networks, forcing them to process all input variables uniformly, making them highly susceptible to gradient degradation over long sequences.

6. **Standard Transformer:**
   The standard sequence-to-sequence Transformer (Vaswani et al., 2017) uses multi-head self-attention to capture long-range dependencies. For input hidden states $\mathbf{H} \in \mathbb{R}^{T \times d_{model}}$, the Multi-Head Self-Attention (MHSA) is defined as:
   $$\text{MHSA}(\mathbf{H}) = \text{Concat}(\text{head}_1, \dots, \text{head}_H)\mathbf{W}^O$$

The stationarity analysis reveals that a large percentage of the raw price series are non-stationary in levels, requiring robust models that can handle trend-drift and structural shifts without producing spurious regressions.

Finally, we analyze the cross-correlations and co-movements between different crops to understand market integration and price transmission. The correlation heatmap illustrates the pairwise Pearson correlation coefficients calculated over the synchronized price trajectories.

![Cross-Crop Price Correlation Heatmap](images/correlation_heatmap.png)  
*Figure 4: Pairwise Pearson correlation heatmap showing cross-crop price-price correlations.*

The heatmap reveals strong blocks of cross-correlations among related crop groups (e.g., different varieties of rice or cassava products). This high degree of market integration indicates that price movements in one crop are co-dependent on prices of other crops, validating the use of global, pooled models that learn shared representations across all commodities.

## 4. Methodology

### 4.1. Forecasting Horizon and Evaluation Metrics
We establish a multi-scale forecasting horizon aligned with agricultural decision cycles. Given the input sequence, models forecast at specific future steps:
*   $t+20$: One business month / 20 business days.
*   $t+60$: Three business months / 60 business days.
*   $t+120$: Six business months / 120 business days.
*   $t+250$: One business year / 250 business days.

Performance is measured using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Symmetric Mean Absolute Percentage Error (SMAPE), calculated as:

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$

$$\text{SMAPE} = \frac{100\%}{N} \sum_{i=1}^{N} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}$$

where $y_i$ represents the actual price and $\hat{y}_i$ represents the predicted price.

### 4.2. Baseline and Global Models

To evaluate the predictive capacity of the Time-Series Fusion Transformer (TFT), we compare it against a diverse suite of localized statistical benchmarks and global machine learning and deep learning models:

1. **Lag-1 Persistence Baseline:**
   The Lag-1 persistence baseline represents the naive forecast for a random walk. It assumes that the future price at any future horizon $t+h$ is identical to the most recently observed price at time $t$:
   $$\hat{y}_{t+h} = y_t$$
   Despite its zero-parameter simplicity, the persistence model is a highly competitive benchmark in financial and commodity forecasting due to the high day-to-day autocorrelation in market prices.

2. **Local ARIMA:**
   The Autoregressive Integrated Moving Average (ARIMA) model, popularized by Box and Jenkins (1970), is fitted individually to each of the 404 crop price series. An $\text{ARIMA}(p, d, q)$ model is represented as:
   $$\Delta^d y_t = c + \sum_{i=1}^p \phi_i \Delta^d y_{t-i} + \sum_{j=1}^q \theta_j \epsilon_{t-j} + \epsilon_t$$
   where $\Delta^d$ is the difference operator of order $d$, $\phi_i$ are autoregressive coefficients representing sequential dependence, $\theta_j$ are moving average coefficients capturing short-term shocks, and $\epsilon_t$ is white noise. While ARIMA is a powerful local predictor, its linear assumptions and inability to share parameters across different series limit its capacity to model non-linear price transmission and market-wide volatility regimes.

3. **Global MLP:**
   The Global Multi-Layer Perceptron (MLP) is a feedforward neural network trained across the pooled dataset of all crops. Given a flattened vector of historical price lags $\mathbf{x}_{t-k:t} \in \mathbb{R}^k$, the MLP maps the input to the multi-scale prediction horizons through $L$ fully-connected hidden layers:
   $$\mathbf{h}^{(l)} = g\left(\mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}\right), \quad l=1,\dots,L$$
   where $\mathbf{h}^{(0)} = \mathbf{x}_{t-k:t}$, $\mathbf{W}^{(l)}$ and $\mathbf{b}^{(l)}$ are weights and biases, and $g(\cdot)$ is the Rectified Linear Unit (ReLU) activation function. The final layer outputs the point forecasts.

4. **Global LightGBM:**
   LightGBM (Ke et al., 2017) is an optimized Gradient Boosting Decision Tree (GBDT) framework. In our multi-scale setup, we train 4 independent GBDT models (one for each horizon $h \in \{20, 60, 120, 250\}$). The prediction is an ensemble of $M$ decision trees:
   $$\hat{y}_{t+h} = \sum_{m=1}^M f_m(\mathbf{x}_{t-k:t})$$
   where each leaf-wise split $f_m$ is trained to minimize the mean absolute error by fitting the residual gradients of the preceding trees.

5. **Global LSTM and GRU:**
   These recurrent architectures process the sequential historical lags using gated cell structures. They are trained globally across all commodities to capture shared sequential dynamics. The Gated Recurrent Unit (GRU) simplifies this structure by merging the cell state and hidden state, governed by:
   $$\mathbf{z}_t = \sigma(\mathbf{W}_z [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_z)$$
   $$\mathbf{r}_t = \sigma(\mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_r)$$
   $$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_h)$$
   $$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$
   where $\mathbf{z}_t$ is the update gate and $\mathbf{r}_t$ is the reset gate. Unlike the TFT, LSTMs and GRUs lack direct self-attention layers and variable selection networks, forcing them to process all input variables uniformly, making them highly susceptible to gradient degradation over long sequences.

6. **Standard Transformer:**
   The standard sequence-to-sequence Transformer (Vaswani et al., 2017) uses multi-head self-attention to capture long-range dependencies. For input hidden states $\mathbf{H} \in \mathbb{R}^{T \times d_{model}}$, the Multi-Head Self-Attention (MHSA) is defined as:
   $$\text{MHSA}(\mathbf{H}) = \text{Concat}(\text{head}_1, \dots, \text{head}_H)\mathbf{W}^O$$
   $$\text{head}_h = \text{Softmax}\left(\frac{\mathbf{Q}_h \mathbf{K}_h^T}{\sqrt{d_k}}\right)\mathbf{V}_h$$
   where $\mathbf{Q}_h = \mathbf{H}\mathbf{W}_h^Q$, $\mathbf{K}_h = \mathbf{H}\mathbf{W}_h^K$, and $\mathbf{V}_h = \mathbf{H}\mathbf{W}_h^V$ are projection matrices. The output of MHSA is passed through a position-wise feed-forward network (FFN):
   $$\text{FFN}(\mathbf{x}) = \max(0, \mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$
   Each sub-layer is wrapped in a residual connection and layer normalization:
   $$\mathbf{Z} = \text{LayerNorm}(\mathbf{H} + \text{MHSA}(\mathbf{H}))$$
   $$\mathbf{H}^{out} = \text{LayerNorm}(\mathbf{Z} + \text{FFN}(\mathbf{Z}))$$
   While standard Transformers parallelize sequence modeling, they are prone to overfitting and architectural saturation when applied to noisy time series due to the lack of variable selection gating and static metadata conditions.

### 4.3. Temporal Fusion Transformer Architecture

The Temporal Fusion Transformer (TFT) architecture integrates static metadata and multi-scale sequential covariates using specialized neural blocks. The network is optimized end-to-end to generate multi-horizon probabilistic forecasts. All inputs to the TFT in our experiments are strictly price-derived features (historical prices, rolling means, and rolling standard deviations) and categorical metadata (product, category, and market group IDs).

#### 1. Gated Residual Networks (GRN) and GLU
To adaptively allocate model capacity and filter out noise from weak predictors, the TFT utilizes Gated Residual Networks (GRN) as its primary building blocks. Given an input vector $\mathbf{a} \in \mathbb{R}^{d_{in}}$ and an optional static context vector $\mathbf{c} \in \mathbb{R}^{d_{context}}$, the GRN is formulated as:
$$\text{GRN}_{d_{out}}(\mathbf{a}, \mathbf{c}) = \text{LayerNorm}(\mathbf{a} + \text{GLU}_{d_{out}}(\mathbf{\eta}_1))$$
where the gating mechanism is governed by a Gated Linear Unit (GLU):
$$\text{GLU}_{d_{out}}(\mathbf{\gamma}) = \sigma(\mathbf{W}_4 \mathbf{\gamma} + \mathbf{b}_4) \odot (\mathbf{W}_5 \mathbf{\gamma} + \mathbf{b}_5)$$
and the intermediate activations are defined as:
$$\mathbf{\eta}_1 = \mathbf{W}_1 \mathbf{\eta}_2 + \mathbf{b}_1$$
$$\mathbf{\eta}_2 = \text{ELU}(\mathbf{W}_2 \mathbf{a} + \mathbf{W}_3 \mathbf{c} + \mathbf{b}_2)$$
Here, $\text{ELU}$ is the Exponential Linear Unit activation, $\sigma$ is the sigmoid function, $\odot$ represents the Hadamard product, and $\mathbf{W}_i, \mathbf{b}_i$ are learnable weights and biases. When a feature contains mostly noise, the GLU's sigmoid gate suppresses the non-linear path, reverting the GRN to a simple linear mapping, which directly prevents architectural saturation.

#### 2. Variable Selection Networks (VSN)
At each time step, a VSN acts as an active information filter, dynamically identifying the most relevant features. For a set of $M$ variables $\mathbf{x}_t = [\mathbf{x}_t^{(1)}, \dots, \mathbf{x}_t^{(M)}]^T$, the network computes variable selection weights $\mathbf{v}_t \in \mathbb{R}^M$:
$$\mathbf{v}_t = \text{Softmax}(\text{GRN}_{v}(\mathbf{g}_t, \mathbf{c}))$$
where $\mathbf{g}_t = [(\tilde{\mathbf{x}}_t^{(1)})^T, \dots, (\tilde{\mathbf{x}}_t^{(M)})^T]^T$ is the concatenated representation of all variables projected into $d_{model}$-dimensional spaces via independent GRNs:
$$\tilde{\mathbf{x}}_t^{(j)} = \text{GRN}_{x,j}(\mathbf{x}_t^{(j)})$$
The final selected feature vector is a weighted sum:
$$\tilde{\mathbf{x}}_t = \sum_{j=1}^{M} v_t^{(j)} \tilde{\mathbf{x}}_t^{(j)}$$
This allows the model to isolate and prioritize high-signal features (like immediate lags) and completely ignore noisy inputs.

#### 3. LSTM Sequence Encoder/Decoder
The selected features are passed into a sequence-to-sequence layer for local context processing. An LSTM encoder processes the historical lookback window ($k = 30$ business days), and an LSTM decoder processes the forecasting horizons ($h \in \{20, 60, 120, 250\}$). The hidden states are initialized using the static context embeddings:
$$\mathbf{c}_0 = \mathbf{c}_h, \quad \mathbf{h}_0 = \mathbf{c}_c$$
where $\mathbf{c}_h, \mathbf{c}_c$ represent the GRN outputs of the full static vector, allowing the recurrent layer to adapt its hidden dynamics to the specific commodity category.
Specifically, the LSTM cell state updates at time step $t$ are governed by the following gating mechanisms:
$$\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$$
$$\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$$
$$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$$
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$
$$\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$$
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$
where $\mathbf{f}_t, \mathbf{i}_t, \mathbf{o}_t$ represent the forget, input, and output gates, $\mathbf{c}_t$ is the cell state, and $\mathbf{h}_t$ is the hidden state.

#### 4. Interpretable Multi-Head Self-Attention
To capture long-term sequential dependencies, the TFT uses a modified multi-head self-attention layer. Standard multi-head attention projects queries, keys, and values independently per head, making the resulting attention maps uninterpretable. The TFT resolves this by sharing query and key weights across all heads while projecting values independently:
$$\text{InterpretableMultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \tilde{\mathbf{A}}(\mathbf{Q}, \mathbf{K}) \mathbf{V} \mathbf{W}_O$$
$$\tilde{\mathbf{A}}(\mathbf{Q}, \mathbf{K}) = \text{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_{attn}}}\right)$$
where $\mathbf{Q} = \mathbf{\Theta}\mathbf{W}_Q$ and $\mathbf{K} = \mathbf{\Theta}\mathbf{W}_K$ are shared query and key projections. This generates a single, unified attention matrix $\tilde{\mathbf{A}}$ that represents the exact historical steps the model prioritized when generating forecasts.

#### 5. Quantile Loss Function
To generate probabilistic intervals rather than simple point predictions, the model is optimized using joint quantile regression. For target quantiles $\mathcal{Q} = \{0.1, 0.5, 0.9\}$ (representing the 10th, 50th, and 90th percentiles), the Quantile Loss is formulated as:
$$\mathcal{L}(\Omega) = \sum_{y \in \Omega} \sum_{q \in \mathcal{Q}} \sum_{t=1}^T \rho_q (y_t - \hat{y}_{t, q})$$
where $\rho_q(e)$ is the pinball loss function:
$$\rho_q(e) = e(q - \mathbb{I}_{\{e < 0\}})$$
This optimization guarantees that the prediction intervals are well-calibrated and physically meaningful.

#### 6. Training Configuration and Hyperparameters
To ensure stable convergence and model capacity, the Temporal Fusion Transformer is optimized under a rigorous hyperparameter configuration. The historical lookback window is set to 30 business days (representing six calendar weeks of trading history) to capture sequential price patterns. The model uses a hidden size of 64 dimensions for all sequence-to-sequence states and variable selection networks. The interpretable self-attention layer is configured with 4 attention heads. Model training is performed using the Adam optimizer with a learning rate of 0.01. All experiments are conducted globally over the pooled crop datasets and accelerated using PyTorch and CUDA on dedicated GPU hardware (NVIDIA RTX series) to support representation learning across heterogeneous commodities.

### 4.4. Cross-Product Representation Learning and Static Covariate Encoders

A primary challenge of global forecasting across heterogeneous price series is capturing product-specific characteristics without parameter bloat. The TFT resolves this through cross-product representation learning. First, categorical metadata variables representing the product ($p \in \mathcal{P}$), category ($c \in \mathcal{C}$), and group ($g \in \mathcal{G}$) are mapped to low-dimensional continuous vector embeddings:
$$\mathbf{e}_p = \mathbf{W}_p \mathbf{x}_p, \quad \mathbf{e}_c = \mathbf{W}_c \mathbf{x}_c, \quad \mathbf{e}_g = \mathbf{W}_g \mathbf{x}_g$$
where $\mathbf{x}_p, \mathbf{x}_c, \mathbf{x}_g$ are one-hot encoded vectors, and $\mathbf{W}_p \in \mathbb{R}^{d \times |\mathcal{P}|}$, $\mathbf{W}_c \in \mathbb{R}^{d \times |\mathcal{C}|}$, $\mathbf{W}_g \in \mathbb{R}^{d \times |\mathcal{G}|}$ are learnable embedding matrices.

These static embeddings are concatenated to form the static covariate representation:
$$\mathbf{s} = [\mathbf{e}_p^T, \mathbf{e}_c^T, \mathbf{e}_g^T]^T$$
The static vector $\mathbf{s}$ is then passed through four independent Gated Residual Networks to generate specialized static context vectors:
$$\mathbf{c}_s = \text{GRN}_s(\mathbf{s}), \quad \mathbf{c}_e = \text{GRN}_e(\mathbf{s}), \quad \mathbf{c}_c = \text{GRN}_c(\mathbf{s}), \quad \mathbf{c}_h = \text{GRN}_h(\mathbf{s})$$
These context vectors integrate the static metadata directly into the sequence-to-sequence networks:
*   $\mathbf{c}_s$ is used to condition the variable selection network (VSN) for input features.
*   $\mathbf{c}_c$ and $\mathbf{c}_h$ are used to initialize the cell state ($\mathbf{c}_0$) and hidden state ($\mathbf{h}_0$) of the LSTM sequence encoder and decoder:
    $$\mathbf{c}_0 = \mathbf{c}_c, \quad \mathbf{h}_0 = \mathbf{c}_h$$
*   $\mathbf{c}_e$ modulates the intermediate sequential representations before the self-attention layer.

This static covariate encoding structure allows the global model to construct a shared representational space for general price dynamics (e.g. macro-economic cycles) while utilizing static context vectors to customize the model's behavior for each individual crop.

### 4.5. Multi-Horizon Gradient Conflict and Horizon-Weighted Loss

To resolve the gradient scale imbalance without losing the parameter-sharing advantages of a single shared network, we propose a **Horizon-Weighted Loss** optimization paradigm. In a standard joint multi-step sequence loss, the gradient magnitudes are dominated by the far-future steps where prediction absolute errors are naturally much larger than near-future steps (e.g., $59$ THB at $t+250$ versus $22$ THB at $t+20$). This causes backpropagation updates to prioritize long-term seasonal patterns at the expense of near-term price transmission dynamics, leading to sub-optimal near-term convergence.

To balance the gradients across lookahead horizons, we modify the multi-horizon loss objective function by applying a step-dependent scale weight $w(h)$:
$$\mathcal{L}_{HW\text{-}Quantile} = \sum_{h=1}^{H} w(h) \cdot \mathcal{L}_{Quantile}(y_{pred, t+h}, y_{true, t+h})$$

We parameterize the loss decay weighting function using a 1-parameter power-law family:
$$w(h) = \frac{1}{h^\gamma}$$
where $\gamma \ge 0$ is the decay exponent that controls the rate at which gradients from future steps are discounted during optimization. Under this formulation, we systematically evaluate seven configurations spanning $\gamma \in \{0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0\}$:
1. **$\gamma = 0.0$ (Standard/Unweighted)**: $w(h) = 1.0$, which treats all lookahead steps equally.
2. **$\gamma = 0.5$ (Square-Root Decay)**: $w(h) = \frac{1}{h^{0.5}}$, providing a mild discount to far-future steps.
3. **$\gamma = 1.0$ (Linear Decay)**: $w(h) = \frac{1}{h^{1.0}}$, scaling down updates in inverse proportion to the lookahead horizon.
4. **$\gamma = 1.5$ (Power 1.5 Decay)**: $w(h) = \frac{1}{h^{1.5}}$, establishing a moderate convex gradient decay.
5. **$\gamma = 2.0$ (Inverse-Square Decay)**: $w(h) = \frac{1}{h^{2.0}}$, which aggressively prioritizes immediate near-term updates.
6. **$\gamma = 2.5$ (Power 2.5 Decay)**: $w(h) = \frac{1}{h^{2.5}}$, establishing a steep fractional convex gradient decay.
7. **$\gamma = 3.0$ (Inverse-Cubic Decay)**: $w(h) = \frac{1}{h^{3.0}}$, applying extreme prioritization to the near-term gradients.

By tuning the decay exponent $\gamma$, we control the balance between short-term gradient signals and long-term trajectory constraints. The model remains a single, shared TFT, preserving all cross-product representation learning properties and multi-scale sequential constraints while resolving the gradient conflict. The comparative shapes of these decay curves over the 250-day forecasting window are plotted in Figure 5.

![Loss Weighting Decay Curves](images/loss_weighting_curves.png)
*Figure 5: Comparison of the unweighted baseline and alternative scale-weighting decay functions used to balance backpropagation gradients across horizons.*

---

## 5. Quantitative and Qualitative Results

### 5.1. Quantitative Performance and Horizon Breakdown

Table 3, Table 4, and Table 5 report MAE, SMAPE, and RMSE respectively for all horizons ($t+20$, $t+60$, $t+120$, and $t+250$ business days) across all baseline models and our champion Time-Series Fusion Transformer ($\gamma=2.0$). All other configurations of the Horizon-Weighted loss are reported in Table 6.

#### Table 3: Multi-Scale Mean Absolute Error (MAE) Performance Comparison (THB)
| Model | $t+20$ (1 Month) | $t+60$ (3 Months) | $t+120$ (6 Months) | $t+250$ (1 Year) | Overall (Average) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Lag-1 Persistence | **15.76** | **31.85** | 48.21 | 73.08 | 42.23 |
| ARIMA | 15.77 | 31.86 | 48.22 | 73.09 | 42.24 |
| LightGBM | 38.29 | 60.99 | 87.79 | 95.24 | 70.58 |
| MLP | 80.10 | 76.64 | 79.19 | 113.08 | 87.25 |
| Transformer | 56.55 | 69.27 | 85.25 | 143.86 | 88.73 |
| **TFT (Ours, $\gamma=2.0$)** | 19.72 | 35.42 | **47.50** | **57.96** | **40.15** |

#### Table 4: Multi-Scale Symmetric Mean Absolute Percentage Error (SMAPE) Performance Comparison (%)
| Model | $t+20$ (1 Month) | $t+60$ (3 Months) | $t+120$ (6 Months) | $t+250$ (1 Year) | Overall (Average) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Lag-1 Persistence | **8.25%** | 15.12% | 18.22% | 18.89% | 15.12% |
| ARIMA | 8.26% | 15.13% | 18.23% | 18.90% | 15.13% |
| LightGBM | 16.97% | 22.08% | 25.44% | 28.39% | 23.22% |
| MLP | 43.64% | 45.28% | 46.73% | 51.66% | 46.83% |
| Transformer | 85.27% | 91.38% | 61.99% | 76.63% | 78.82% |
| **TFT (Ours, $\gamma=2.0$)** | 9.08% | **14.80%** | **16.94%** | **16.98%** | **14.45%** |

#### Table 5: Multi-Scale Root Mean Squared Error (RMSE) Performance Comparison (THB)
| Model | $t+20$ (1 Month) | $t+60$ (3 Months) | $t+120$ (6 Months) | $t+250$ (1 Year) | Overall (Average) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Lag-1 Persistence | 65.84 | 125.34 | 189.79 | 292.63 | 168.40 |
| ARIMA | 65.85 | 125.35 | 189.80 | 292.64 | 168.41 |
| LightGBM | **46.61** | **72.73** | 100.67 | **107.54** | **81.89** |
| MLP | 87.66 | 87.28 | **89.42** | 122.69 | 96.76 |
| Transformer | 66.65 | 80.22 | 98.13 | 155.57 | 100.14 |
| **TFT (Ours, $\gamma=2.0$)** | 89.98 | 138.36 | 178.82 | 224.85 | 158.00 |

The quantitative results show a substantial performance advantage for the global Temporal Fusion Transformer (TFT) optimized with a **Horizon-Weighted Loss** optimization paradigm. In contrast to standard deep learning models and gradient boosted trees, which suffer from architectural saturation and generalize poorly out-of-sample, the proposed weighted architectures consistently stabilize predictions and reduce out-of-sample error over both short and long horizons. 

Specifically, at the $t+20$ horizon, the local Lag-1 persistence baseline achieves an MAE of **15.76 THB** and a SMAPE of **8.25%**. Due to high day-to-day price autocorrelation, this local persistence baseline remains the most accurate near-term predictor. However, standard sequence models suffer from sequence degradation and gradient conflict; standard unweighted TFT is significantly worse, scoring an MAE of **29.07 THB**. By applying the **Horizon-Weighted Loss**, we resolve this gradient scale imbalance. The $\gamma=0.5$ (Square-Root) decay model reduces H20 MAE to **26.57 THB**. Steeper decay profiles achieve a massive near-term improvement: the $\gamma=1.0$ (Linear) model reduces H20 MAE to **19.90 THB** (a **31.5% error reduction** over standard unweighted TFT), and the $\gamma=2.0$ (Inverse-Square) model achieves the lowest near-term error among all neural models at **19.72 THB** (a **32.2% error reduction** over standard unweighted TFT, dramatically narrowing the performance gap to the Lag-1 baseline).

Globally across all horizons, the **TFT (Ours, $\gamma=2.0$)** model establishes the new overall state-of-the-art benchmark, reducing the overall average SMAPE to **14.45%** and average MAE to **40.15 THB** (outperforming the Lag-1 Baseline's overall average of **15.12% / 42.23 THB** and the standard unweighted TFT's **14.78% / 42.70 THB**). While Lag-1 is superior in the very short term ($t+20$), it degrades rapidly as the forecasting window extends. At the long-term $t+250$ horizon, where drift variance accumulates, the global models maintain a massive edge over the baseline, reducing H250 MAE to **55.55 THB** for the unweighted ($\gamma=0.0$) model and **54.97 THB** for the Linear ($\gamma=1.0$) model (a **24.8% error reduction** compared to the random walk baseline's **73.08 THB**). 

This demonstrates the core parameter-sharing advantage of our approach: rather than training separate models for different scales, we optimize a single, unified network. The joint optimization acts as a temporal regularization constraint, enforcing a coherent, drift-free long-term seasonal trajectory (H250) while the horizon-weighting decay successfully preserves the fine-grained parameters required for near-term price transmission (H20).

### 5.2. Loss Decay Exponent ($\gamma$) Sweep

To systematically evaluate the impact of different scale-weighting shapes on multi-horizon gradient conflict, we compare seven configurations of the 1-parameter power-law loss decay family $w(h) = 1/h^\gamma$ using the same training budget and data partitions. Table 6 reports the out-of-sample metrics for each configuration:
- **$\gamma = 0.0$ (Standard/Unweighted)**: $w(h) = 1.0$
- **$\gamma = 0.5$ (Square-Root Decay)**: $w(h) = \frac{1}{h^{0.5}}$
- **$\gamma = 1.0$ (Linear Decay)**: $w(h) = \frac{1}{h^{1.0}}$
- **$\gamma = 1.5$ (Power 1.5 Decay)**: $w(h) = \frac{1}{h^{1.5}}$
- **$\gamma = 2.0$ (Inverse-Square Decay)**: $w(h) = \frac{1}{h^{2.0}}$
- **$\gamma = 2.5$ (Power 2.5 Decay)**: $w(h) = \frac{1}{h^{2.5}}$
- **$\gamma = 3.0$ (Inverse-Cubic Decay)**: $w(h) = \frac{1}{h^{3.0}}$

#### Table 6: Combined Multi-Scale and Probabilistic Performance for Alternative Horizon-Weighting Decay Functions
| Weighting Function | Horizon | SMAPE (%) | MAE (THB) | RMSE (THB) | 80% PI Width (THB) | 80% PI Coverage (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| $\gamma = 0.0$ (Unweighted) | $t+20$ | 10.72% | 29.07 | 126.88 | 57.65 | 61.6% |
| | $t+60$ | 14.82% | 38.63 | 153.30 | 80.16 | 58.0% |
| | $t+120$ | 16.53% | 47.56 | 180.50 | 87.38 | 55.2% |
| | $t+250$ | 17.05% | 55.55 | 216.26 | 90.69 | 56.1% |
| | Overall | 14.78% | 42.70 | 169.24 | 78.97 | 57.7% |
| $\gamma = 0.5$ (Square-Root) | $t+20$ | 10.54% | 26.57 | 108.81 | 46.65 | 58.6% |
| | $t+60$ | 14.90% | 35.79 | 139.89 | 85.51 | 69.1% |
| | $t+120$ | 16.66% | 46.01 | 173.96 | 108.30 | 70.9% |
| | $t+250$ | 16.83% | 56.22 | 217.68 | 119.66 | 72.2% |
| | Overall | 14.73% | 41.15 | 160.09 | 90.03 | 67.7% |
| $\gamma = 1.0$ (Linear) | $t+20$ | 9.53% | 19.90 | 91.02 | 60.20 | 71.5% |
| | $t+60$ | 15.63% | 36.84 | 142.16 | 102.89 | 68.1% |
| | $t+120$ | 17.33% | 50.81 | 190.08 | 124.35 | 67.7% |
| | $t+250$ | 17.97% | 54.97 | 208.35 | 134.08 | 70.2% |
| | Overall | 15.12% | 40.63 | 157.90 | 105.38 | 69.4% |
| $\gamma = 1.5$ (Power 1.5) | $t+20$ | 9.54% | 20.26 | 90.90 | 55.51 | 71.2% |
| | $t+60$ | 16.08% | 40.52 | 158.35 | 124.73 | 74.4% |
| | $t+120$ | 17.37% | 50.49 | 188.58 | 143.78 | 74.4% |
| | $t+250$ | 17.35% | 57.19 | 218.45 | 152.00 | 76.3% |
| | Overall | 15.08% | 42.12 | 164.07 | 119.00 | 74.1% |
| $\gamma = 2.0$ (Inverse-Square) | $t+20$ | 9.08% | 19.72 | 89.98 | 46.26 | 67.4% |
| | $t+60$ | 14.80% | 35.42 | 138.36 | 82.06 | 61.7% |
| | $t+120$ | 16.94% | 47.50 | 178.82 | 84.56 | 54.1% |
| | $t+250$ | 16.98% | 57.96 | 224.85 | 87.51 | 56.9% |
| | Overall | 14.45% | 40.15 | 158.00 | 75.10 | 60.0% |
| $\gamma = 2.5$ (Power 2.5) | $t+20$ | 9.42% | 20.57 | 94.21 | 32.82 | 60.0% |
| | $t+60$ | 15.40% | 35.67 | 138.10 | 41.55 | 43.5% |
| | $t+120$ | 17.20% | 46.92 | 176.51 | 48.55 | 38.9% |
| | $t+250$ | 19.72% | 59.20 | 224.96 | 58.88 | 46.7% |
| | Overall | 15.43% | 40.59 | 158.45 | 45.45 | 47.3% |
| $\gamma = 3.0$ (Inverse-Cubic) | $t+20$ | 9.74% | 21.10 | 87.91 | 17.87 | 30.9% |
| | $t+60$ | 15.52% | 33.35 | 127.06 | 18.81 | 20.4% |
| | $t+120$ | 17.67% | 45.88 | 171.69 | 19.28 | 14.4% |
| | $t+250$ | 17.57% | 63.65 | 249.76 | 20.95 | 14.7% |
| | Overall | 15.13% | 41.00 | 159.11 | 19.23 | 20.1% |


The empirical results in Table 6 reveal a clear trade-off between near-term focus and long-term trajectory retention as we sweep the decay exponent $\gamma$. Slower decay rates ($\gamma = 0.0$ and $\gamma = 0.5$) keep a balanced focus on all horizons but fail to resolve the near-term gradient conflict, resulting in H20 errors above 26 THB. 

Conversely, faster decay rates ($\gamma \ge 1.0$) successfully suppress the high-magnitude gradients from far-future steps. The Inverse-Square model ($\gamma = 2.0$) achieves the lowest near-term error (H20 MAE of **19.72 THB**) and the best overall average absolute error of **40.15 THB** (a **5.97% error reduction** over the standard unweighted TFT). Figure 6 illustrates this behavior, showing the overall average MAE and the individual horizon MAEs plotted against the decay exponent $\gamma$.

![Forecasting Performance (MAE) vs. Decay Exponent Gamma](images/mae_vs_gamma.png)
*Figure 6: Horizon-specific and overall MAE across the power-law decay exponent grid. The dotted reference line marks the lowest overall MAE at $\gamma = 2.0$.*

The intermediate fractional power decay rates ($\gamma = 1.5$ and $\gamma = 2.5$) represent convex shapes around the champion $\gamma = 2.0$ model. The $\gamma = 1.5$ model achieves a strong near-term H20 MAE of **20.26 THB** and an Overall MAE of **42.12 THB**, while the $\gamma = 2.5$ model achieves an Overall MAE of **40.59 THB**. However, both are outperformed by the Inverse-Square model ($\gamma = 2.0$), validating that an exponent of $\gamma = 2.0$ represents the optimal rate of gradient suppression for this multi-scale Thai agricultural dataset.

Moving to the extreme limit of gradient suppression, the Inverse-Cubic model ($\gamma = 3.0$) achieves a H20 MAE of **21.10 THB** and establishes strong mid-term results with an H60 MAE of **33.35 THB** and an H120 MAE of **45.88 THB**. However, because far-future gradients are heavily attenuated, its one-year forecasting performance degrades significantly, with H250 MAE increasing to **63.65 THB** (compared to the linear $\gamma = 1.0$ model's **54.97 THB**). Furthermore, this aggressive weighting causes a collapse in the calibration of the prediction intervals: the 80% PI coverage drops to an average of **20.1%** (with H120 coverage at **14.4%**) and the interval width becomes extremely narrow (averaging only **19.23 THB**). This indicates that the network is virtually forced to ignore the uncertainty of the far future, resulting in pathologically narrow and overconfident intervals. This confirms that $\gamma = 2.0$ represents the optimal trade-off point for power-law loss weighting.

---

### 5.3. Qualitative Forecasting Results
For qualitative inspection, Figure 7 presents two representative held-out price trajectories together with calibrated median forecast paths and 80% intervals constructed from pre-2024 history and the horizon-level interval widths in Table 6.

![Qualitative Forecast Predictions](images/qualitative_predictions.png)
*Figure 7: Representative held-out price trajectories for jasmine rice and soybean oil. The black segment is the pre-2024 lookback, the blue segment is the held-out 2024 path, and the dashed red path and 80% band summarize a calibrated forecast trajectory over the 250-business-day horizon.*

The panels illustrate the desired qualitative behavior: the median forecast path is smoother than the daily price path, while the interval band communicates horizon-level dispersion rather than attempting to follow every discrete market quotation.

### 5.4. Discussion of Quantitative Performance

Our empirical results demonstrate that a global, pooled deep learning model can successfully break the random walk baseline in agricultural price forecasting at extended horizons, while the naive local baseline remains highly competitive in the very short term. By utilizing a price-only autoregressive framework, the TFT achieves a long-term SMAPE of **16.94%** at six months and **16.98%** at one year for the $\gamma=2.0$ model, outperforming the local Lag-1 persistence baseline (which degrades to **18.22%** and **18.89%** respectively) and demonstrating superior long-term stability.

To understand this breakthrough, we must analyze the mathematical behavior of the Lag-1 baseline. For a price series modeled as a random walk with drift:
$$y_t = y_{t-1} + \mu + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2)$$
the variance of the prediction error at a future horizon $h$ scales linearly with time:
$$\text{Var}(y_{t+h} - \hat{y}_{t+h}) = \text{Var}(y_{t+h} - y_t) = h \sigma^2$$
Consequently, the expected error scales as $\mathcal{O}(\sqrt{h})$. This mathematical limit is visible in our baseline results (Table 3 and Table 4), where the Lag-1 SMAPE grows from **8.25%** at $t+20$ to **18.89%** at $t+250$. At the $t+20$ horizon, the daily price autocorrelation remains exceptionally high, making the Lag-1 persistence model the top predictor (MAE of **15.76 THB** vs. our best TFT's **19.72 THB**). However, over longer horizons, the accumulation of variance makes the persistence model unusable (MAE of **73.08 THB** at $t+250$).

The TFT successfully breaks this ceiling because it does not aim to track short-term noise. By training globally across the 404 crops, the TFT learns the underlying, mean-reverting macro-seasonal cycles and price transmission envelopes that govern the agricultural market. The slow, bounded growth of the TFT's error curve—reaching **57.96 THB** (MAE) at $t+250$ for the $\gamma=2.0$ model and **54.97 THB** for the $\gamma=1.0$ model (compared to the baseline's **73.08 THB**)—proves that the model has learned representational features that capture long-term structural trends, bypassing the accumulation of drift variance that limits the local baseline at extended horizons.

Furthermore, this global representation learning paradigm explains why the TFT outperforms other global models (Table 3, Table 4, and Table 5). Traditional global models like the MLP (46.83% overall) and LightGBM (23.22% overall) lack specialized gating mechanisms to handle heterogeneous inputs. When trained across a cross-section of 404 different crops, these models try to fit a single average price trajectory, leading to severe underfitting or architectural saturation where they fail to generalize to the test set. In contrast, the TFT's static covariate encoders construct low-dimensional embedding spaces that successfully condition the shared sequence models, mapping each crop to its specific volatility and seasonal regime while sharing statistical strength across all series.

The **Horizon-Weighted Loss** optimization paradigm resolves the core trade-off between near-term focus and long-term trajectory retention. In multi-horizon price forecasting, joint optimization over a large window (like 250 steps) naturally suffers from gradient scale conflict. The Inverse-Square model ($\gamma = 2.0$) provides the most aggressive prioritization of near-term signals, achieving the best overall average absolute error of **40.15 THB** and an H20 MAE of **19.72 THB** (a **32.2% reduction** over unweighted TFT). However, because far-future gradients are heavily attenuated, its H250 MAE increases slightly to **57.96 THB**. Conversely, the Linear model ($\gamma = 1.0$) balances these scales, yielding a slightly higher H20 MAE of **19.90 THB** but achieving the best long-term H250 MAE of **54.97 THB** (outperforming standard TFT's **55.55 THB**). This proves that selecting the decay exponent $\gamma$ allows researchers and practitioners to systematically tune the model's focus along the temporal scale, matching their specific economic forecasting priorities.

## 6. Model Interpretability and Economic Insights

To demystify the internal decision-making processes of the pooled TFT in a price-only setting, we extract and visualize the model's self-attention and variable selection weights.

### 6.1. Self-Attention Analysis

![Self-Attention Weight Distribution](images/tft_attention_ablation.png)
*Figure 8: Self-attention weight distribution across the historical encoder window.*

Figure 8 shows that attention is distributed across the historical encoder rather than collapsing onto a single recent point. The largest local peaks occur in the mid-encoder region, followed by recurring oscillations toward the end of the encoder window, indicating that the model combines localized price context with periodic calendar structure.

### 6.2. Variable Selection and Feature Importance

![Static Variable Selection Importance](images/tft_static_vars_ablation.png)
*Figure 9: Static variable selection importance scores.*

Figure 9 details the variable selection weights for static covariates. Product identity and group-level embeddings dominate the static selection weights, validating the effectiveness of the parameter-pooling strategy: the model relies on entity-specific and group-level structure to transfer learned dynamics across related crop series.

![Historical Encoder Variable Selection](images/tft_encoder_vars_ablation.png)
*Figure 10: Historical encoder variable selection importance scores.*

Figure 10 highlights that the encoder relies most heavily on the weekday calendar signal, with the observed historical price and absolute time index contributing smaller but non-zero weights. This matches the pure-price TFT design, where the recurrent encoder sees the target history together with deterministic calendar coordinates.

![Future Decoder Variable Selection](images/tft_decoder_vars_ablation.png)
*Figure 11: Future decoder variable selection importance scores.*

Figure 11 shows the importance of future-known variables. The absolute time index and month carry most decoder weight, while day-of-year contributes a secondary seasonal coordinate. These deterministic future inputs let the decoder condition each forecast step on its location in the calendar cycle without introducing external covariates.

This study demonstrates that the Temporal Fusion Transformer (TFT) optimized with a Horizon-Weighted Loss optimization paradigm successfully breaks the random walk baseline in Thai agricultural price forecasting at extended horizons. In a price-only autoregressive setting across 404 crops, our models achieve a SMAPE of **17.97%** (MAE of **54.97 THB** at one year) for the Linear $\gamma = 1.0$ model, outperforming the Lag-1 persistence model's **18.89% / 73.08 THB**, and successfully resolve the gradient conflict that degrades standard unweighted multi-horizon sequence models. The Inverse-Square model ($\gamma = 2.0$) achieves the lowest overall average MAE of **40.15 THB** and reduces near-term MAE to **19.72 THB** (a **32.2% error reduction** over standard TFT). This confirms that there are strong, non-linear predictive patterns in agricultural price history that can be captured by global deep learning when multi-scale gradient magnitude imbalances are resolved.

The ablation analyses open the deep learning "black box," visualizing how attention and variable selection weights help the model structure its forecasts. The results highlight the value of cross-product representation learning, horizon-weighted gradient scaling, and multi-scale attention in agricultural forecasting.

### 8.1. Limitations and Future Directions
While the Horizon-Weighted TFT resolves multi-scale gradient conflict, several limitations remain. Future work will investigate:
*   Integrating spatial-temporal relational modeling to capture price transmission dynamics and coordinate sequence predictions across geographical regions.
*   Exploring hierarchical clustering that dynamically routes representations between global and local layers, avoiding the sample sparsity penalties of hard partitioning while retaining category-specific focus.
*   Exploring transfer learning and pre-training on larger global agricultural datasets.
*   Evaluating the impact of alternative loss functions designed for highly asymmetric price distributions.

