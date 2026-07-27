# Cover Letter

27 July 2026

The Editor-in-Chief
*Knowledge-Based Systems*
Elsevier

**Re: Submission of "Beyond the Random Walk: Horizon-Weighted Temporal Fusion Transformers for Thai Agricultural Commodity Price Forecasting"**

Dear Editor,

We submit our manuscript, "Beyond the Random Walk: Horizon-Weighted Temporal Fusion Transformers for Thai Agricultural Commodity Price Forecasting," for consideration in *Knowledge-Based Systems*. The manuscript is original, has not been published previously, and is not under consideration elsewhere.

Agricultural commodity price forecasting is a consequential application of intelligent decision support, but non-stationarity and low signal-to-noise ratios often make complex models less accurate than persistence. We study this failure mode on eight years of daily prices for 404 Thai crop products from the Ministry of Commerce, Department of Internal Trade.

We identify a multi-scale gradient imbalance in pooled multi-horizon training and introduce a **Horizon-Weighted Quantile Loss**, $w(h)=1/h^\gamma$, for the Temporal Fusion Transformer. We evaluate all seventeen half-step values of $\gamma$ from 0 to 8 and select $\gamma=4.5$ using a held-out validation year. On matched 2024--2025 test windows, the selected model reduces aggregate MAE by 12.0% and one-year MAE by 36.8% against Lag-1 persistence. We also disclose the short-horizon deficit and 57.7% per-crop win rate, test product-clustered significance, replicate results by test year and training seed, compare with classical, neural, and zero-shot foundation-model baselines, and calibrate 80% prediction intervals using validation data.

The manuscript fits *Knowledge-Based Systems* because it contributes a focused learning objective for a documented failure mode, uses pooled entity and group representations to transfer structure across heterogeneous commodities, and reports interpretable variable-selection and attention analyses. The forecast horizons and calibrated intervals are designed for agricultural decision support rather than accuracy ranking alone. The supplementary material gives model-configuration provenance and an equal-budget, validation-based search for every learned reference class.

All authors have approved the submission and declare no conflicts of interest. We believe the paper offers both a general methodological contribution and a reproducible benchmark for an under-studied developing-market commodity dataset.

Thank you for your consideration. We look forward to your response.

Sincerely,

Kritaphat Songsri-in
Department of Computer Science, Faculty of Science and Technology
Nakhon Si Thammarat Rajabhat University
1 Moo 4, Tha Ngio, Mueang Nakhon Si Thammarat
Nakhon Si Thammarat 80280, Thailand
kritaphat_son@nstru.ac.th
On behalf of all authors.
