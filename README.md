# Academic Paper Revision Package: Crop Price Forecasting

This folder is a self-contained package designed for an AI assistant to review, verify, and revise our academic manuscript on multi-scale crop price forecasting.

## Package Directory Structure

- **`paper/final_paper.md`**: The primary academic manuscript file. Open and edit this file to perform revisions.
- **`paper/images/`**: Contains manuscript figure files; `final_paper.md` must embed exactly 11.
- **`src/`**: The core forecasting pipeline source code (data loading, preprocessing, feature engineering, and model training).
- **`results/`**: Standardized quantitative evaluation results (`metrics.csv`, `summary_report.md`, and `exhaustive_metrics_with_da.csv`).
- **`verify_keywords.py`**: Compliance script checking for negative constraints and image counts.
- **`verify_experiments.py`**: E2E pipeline verification test suite.
- **`PROJECT.md`**: Project milestone definitions and interface contracts.

---

## Verification Commands

To check the compliance and validity of any edits you make to the paper or code, run the following verification steps from the root of this folder:

1. **Verify Paper Constraints (Negative Keywords & Images)**:
   ```bash
   conda run -n base python verify_keywords.py
   ```
   *Ensures the paper contains zero forbidden weather terms (weather, rain, temperature, climate, precipitation) and exactly 11 embedded images.*

2. **Verify Code and Experiment Pipeline (88 E2E Test Cases)**:
   ```bash
   conda run -n base python verify_experiments.py
   ```
   *Verifies that all baseline and deep learning models train, validate, and compute metrics correctly without target leakage or numerical errors.*

---

## Documented Results and Narrative

When editing `paper/final_paper.md`, make sure you preserve the following key findings and constraints:

1. **Short-Term Horizon ($t+20$)**:
   - The naive local **Lag-1 Persistence baseline** remains the top predictor (MAE of **15.76 THB** / SMAPE of **8.25%**) due to high daily autocorrelation.
   - Standard unweighted sequence models (TFT $\gamma=0.0$) score a much higher MAE of **29.07 THB**.
   - Our proposed **Horizon-Weighted Loss** with Inverse-Square decay ($\gamma=2.0$) achieves a massive near-term improvement, reducing MAE to **19.72 THB** (a **32.2% error reduction** over standard TFT), significantly bridging the gap to the persistence baseline.

2. **Long-Term Horizon ($t+250$)**:
   - Random walk drift variance accumulates as $\mathcal{O}(\sqrt{h})$, causing the Lag-1 baseline error to spike to **73.08 THB** at one year.
   - The global TFT successfully models long-term macro-seasonal trends, with the Linear decay model ($\gamma=1.0$) achieving the best one-year MAE of **54.97 THB** (a **24.8% error reduction** compared to the baseline).

3. **Overall Average**:
   - The champion Inverse-Square model ($\gamma = 2.0$) achieves the lowest overall average MAE of **40.15 THB** across all horizons, outperforming both the Lag-1 baseline (**42.23 THB**) and the standard unweighted TFT (**42.70 THB**).
