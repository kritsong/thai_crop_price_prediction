"""Generate paper/supplementary.tex (and compile to PDF) from the reference
hyperparameter search artifacts. Run AFTER run_reference_search.py.

House style rules enforced here as in the main paper. No em dashes, no
'champion', no prose elaboration colons.
"""
import sys, io, json, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pandas as pd

from paths import ROOT as _ROOT

ROOT = str(_ROOT)
search = pd.read_csv(os.path.join(ROOT, "results", "reference_hparam_search.csv"))
final = pd.read_csv(os.path.join(ROOT, "results", "reference_hparam_final.csv"))
multiseed = pd.read_csv(
    os.path.join(ROOT, "results", "champion_gamma45_multiseed_metrics.csv")
)
eda = pd.read_csv(os.path.join(ROOT, "results", "eda_product_summary.csv"))
gamma_sweep = pd.read_csv(
    os.path.join(ROOT, "results", "gamma_sweep_summary.csv")
)

ORIGINAL_TEST = {  # Table 5 of the main paper (overall = mean of the four horizons)
    "lightgbm": 44.83, "mlp": 56.88, "lstm": 59.59, "transformer": 70.82,
}
FAMILY_NAMES = {"lightgbm": "Global LightGBM", "mlp": "Global MLP",
                "lstm": "Global LSTM", "transformer": "Global Transformer"}
PARAM_NAMES = {
    "num_leaves": "leaves", "max_depth": "depth", "min_data_in_leaf": "min/leaf",
    "learning_rate": "lr", "feature_fraction": "ff", "bagging_fraction": "bf",
    "bagging_freq": "freq", "h1": "$h_1$", "h2": "$h_2$", "drop": "dropout",
    "lr": "lr", "hidden": "hidden", "d_model": "$d_{model}$", "layers": "layers",
    "nhead": "heads",
}

def fmt_params(js):
    d = json.loads(js)
    return ", ".join(f"{PARAM_NAMES.get(k, k)} {v}" for k, v in d.items())

def family_table(fam):
    sub = search[search.family == fam].sort_values("config_id")
    best_id = int(sub.loc[sub.val_mae_mean.idxmin(), "config_id"])
    rows = []
    for _, r in sub.iterrows():
        tag = ""
        if r.published and r.config_id == best_id:
            tag = " (original, selected)"
        elif r.published:
            tag = " (original)"
        elif r.config_id == best_id:
            tag = " (selected)"
        bold = lambda s: f"\\textbf{{{s}}}" if r.config_id == best_id else s
        rows.append(
            f"{int(r.config_id)}{tag} & {fmt_params(r.params)} & "
            + " & ".join(bold(f"{r[f'val_mae_{h}']:.2f}") for h in [20, 60, 120, 250])
            + f" & {bold(f'{r.val_mae_mean:.2f}')} \\\\"
        )
    return "\n".join(rows), best_id

def esc(s):
    return s

tex = []
tex.append(r"""\documentclass[11pt]{article}
\usepackage[margin=2.7cm]{geometry}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\graphicspath{{images/}}
\usepackage[hidelinks]{hyperref}
\setlength{\emergencystretch}{2em}
\hypersetup{pdftitle={Supplementary Material: Horizon-Weighted Temporal Fusion Transformers for Thai Agricultural Commodity Price Forecasting},pdfauthor={Kritaphat Songsri-in, Auyporn Chukeaw, Munlika Rattaphun, Walaiporn Sornkliang, Rattayagon Thaiphan}}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\renewcommand{\thesection}{S\arabic{section}}
\title{Supplementary Material for ``Beyond the Random Walk: Horizon-Weighted Temporal Fusion Transformers for Thai Agricultural Commodity Price Forecasting''}
\author{Kritaphat Songsri-in, Auyporn Chukeaw, Munlika Rattaphun,\\
Walaiporn Sornkliang, and Rattayagon Thaiphan}
\date{}
\begin{document}
\maketitle

\section{Fairness protocol for reference-model configuration}
A recurring concern in comparative forecasting studies is that the proposed model receives careful tuning while reference models run at defaults, biasing the comparison. This supplement documents the configuration provenance of every learned model in the study and reports a hyperparameter search for each learned reference class of Table~5 under one common protocol.

\begin{itemize}
\item \textbf{Matched architecture-search budgets.} The TFT architecture was informed by a 10-trial exploratory search (Section~S2) and held constant across all seventeen loss exponents. Each learned reference class receives the same 10-configuration architecture or hyperparameter budget (Section~S3), and the configuration used in the main paper is always one candidate. The separate seventeen-point loss sweep is the paper's primary method ablation and is disclosed in full; the total tuning budgets are therefore not identical.
\item \textbf{Validation-only selection.} Every candidate is fitted on labels within 2018--2022 only and scored by matched-window MAE on validation windows whose horizon-$h$ label falls inside 2023 (12{,}120 windows at $t{+}20$, 28{,}280 at $t{+}60$, 52{,}520 at $t{+}120$, and 105{,}040 at $t{+}250$ over the 404 crops), averaged over $h \in \{20, 60, 120, 250\}$. No test-period information enters any selection.
\item \textbf{Identical final conditions.} The validation-selected configuration is then refitted on labels through 2023, the same training window the published reference models used, and evaluated on the identical 2024--2025 matched test windows as Table~5 of the main paper (110{,}292 windows per horizon, 404 crops).
\end{itemize}

\section{TFT architecture provenance}
The TFT architecture used throughout the main paper (hidden size 64, 4 attention heads, continuous-variable encoding size 8, dropout 0.1, learning rate 0.01) was fixed before the decay-exponent sweep and was informed by a 10-trial exploratory Optuna search (median pruner, 3-epoch budget per trial) under the unweighted quantile objective. Table~\ref{tab:tftspace} lists the search space. The historical trial ledger was not retained, so this is architecture provenance rather than a reproducible claim that these values were the unique validation optimum. The architecture was then held constant across all seventeen decay exponents of the main experiment. The exponent was swept exhaustively over $\gamma \in \{0.0, 0.5, \dots, 8.0\}$, with full per-horizon results disclosed in Table~6 of the main paper.

\begin{table}[htbp]\centering
\caption{Search space of the 10-trial TFT architecture search.}
\label{tab:tftspace}
\small
\begin{tabular}{ll}
\toprule
Hyperparameter & Space \\
\midrule
Hidden state size & $\{16, 32, 64\}$ \\
Attention heads & $\{1, 2, 4\}$ \\
Dropout & $[0.1, 0.3]$ \\
Continuous-variable encoding size & $\{8, 16, 32\}$ \\
Learning rate & log-uniform $[10^{-3}, 10^{-1}]$ \\
\bottomrule
\end{tabular}
\end{table}

\section{Reference-model search spaces and validation results}
Tables~\ref{tab:lgb}--\ref{tab:tf} report all ten candidate configurations per class with their per-horizon validation MAE. Configuration~0 is always the original configuration used in the main paper. The gradient-boosting search varies tree complexity, regularization, and sampling; the MLP search varies width, dropout, and learning rate around the original values; the LSTM search varies hidden-state size and learning rate; the Transformer search varies model width, depth, head count, and learning rate. Training budgets (boosting rounds, epochs, batch sizes) are held at their original values within each class so that the search isolates configuration rather than compute. The GRU variant of the recurrent class is omitted from the search for the reason given in Section~4.3 of the main paper; it is not part of Table~5 and does not outperform the naive baseline overall.
""")

for fam, label in [("lightgbm", "tab:lgb"), ("mlp", "tab:mlp"), ("lstm", "tab:lstm"), ("transformer", "tab:tf")]:
    body, best_id = family_table(fam)
    tex.append(
        "\\begin{table}[htbp]\\centering\n"
        f"\\caption{{{FAMILY_NAMES[fam]} search. Validation MAE (THB) per horizon; the validation-selected configuration is bolded.}}\n"
        f"\\label{{{label}}}\n\\scriptsize\n\\setlength{{\\tabcolsep}}{{3.5pt}}\n"
        "\\begin{tabular}{llccccc}\n\\toprule\n"
        "Cfg & Parameters & $t{+}20$ & $t{+}60$ & $t{+}120$ & $t{+}250$ & Mean \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )

# final comparison table (selected vs rerun-original vs Table 5)
rows = []
for fam in ["lightgbm", "mlp", "lstm", "transformer"]:
    sel = final[(final.family == fam) & (final.which == "selected")].iloc[0]
    pub = final[(final.family == fam) & (final.which == "published")].iloc[0]
    same = int(sel.config_id) == int(pub.config_id)
    rows.append(
        f"{FAMILY_NAMES[fam]} & {int(sel.config_id)}{' (= original)' if same else ''} & "
        + " & ".join(f"{sel[f'test_mae_{h}']:.2f}" for h in [20, 60, 120, 250])
        + f" & {sel.test_mae_mean:.2f} & {pub.test_mae_mean:.2f} & {ORIGINAL_TEST[fam]:.2f} \\\\"
    )
baseline_test = final.baseline_test_mae.iloc[0]

tex.append(
    "\\section{Test-set impact of validation-selected configurations}\n"
    "Table~\\ref{tab:impact} reports the 2024--2025 matched-window test MAE of each validation-selected configuration after refitting on labels through 2023. The ``original (rerun)'' column refits configuration~0 inside this supplement's pipeline under the same random initialization as the search candidates, and the ``main paper'' column repeats the corresponding overall value of Table~5. For the gradient-boosting class the rerun reproduces the main-paper value essentially exactly, which validates the pipeline; for the neural classes the rerun differs from the main-paper value by several THB in either direction under an identical configuration, illustrating initialization variance. Tuning-attributable differences should therefore be read against the rerun column rather than the main-paper column. The matched Lag-1 persistence baseline scores "
    f"{baseline_test:.2f}"
    " THB on the same windows.\n\n"
    "\\begin{table}[htbp]\\centering\n"
    "\\caption{Test MAE (THB) of validation-selected reference configurations against the original Table~5 configurations.}\n"
    "\\label{tab:impact}\n\\small\n"
    "\\begin{tabular}{llccccccc}\n\\toprule\n"
    " & & \\multicolumn{5}{c}{Selected configuration} & Original & Main \\\\\n"
    "Model class & Cfg & $t{+}20$ & $t{+}60$ & $t{+}120$ & $t{+}250$ & Overall & (rerun) & paper \\\\\n\\midrule\n"
    + "\n".join(rows)
    + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
)

# selected-exponent seed sensitivity
seed_rows = []
seed_subset = multiseed[
    multiseed["Seed"].isin([42, 43, 44])
    & multiseed["Horizon"].isin([20, 60, 120, 250])
].copy()
for seed, sub in seed_subset.groupby("Seed", sort=True):
    sub = sub.set_index("Horizon")
    values = [float(sub.loc[h, "MAE"]) for h in [20, 60, 120, 250]]
    seed_rows.append(
        f"{int(seed)} & "
        + " & ".join(f"{value:.2f}" for value in values)
        + f" & {sum(values) / len(values):.2f} \\\\"
    )
baseline = seed_subset.groupby("Horizon", sort=True)["Baseline_MAE"].first()
baseline_values = [float(baseline.loc[h]) for h in [20, 60, 120, 250]]
seed_rows.append(
    "Persistence & "
    + " & ".join(f"{value:.2f}" for value in baseline_values)
    + f" & {sum(baseline_values) / len(baseline_values):.2f} \\\\"
)
tex.append(
    "\\section{Selected-exponent seed sensitivity}\n"
    "Table~\\ref{tab:seeds} reports independent retraining of the selected "
    "$\\gamma=4.5$ configuration. Exponent selection itself used the common "
    "seed-42 sweep and 2023 validation MAE; seeds 43 and 44 are post-selection "
    "sensitivity runs. All three runs beat persistence overall and at "
    "$t{+}120$ and $t{+}250$, while all remain worse at $t{+}20$. This supports "
    "the qualitative horizon pattern but also shows that the exact overall "
    "point estimate varies from 39.76 to 43.70 THB.\n\n"
    "\\begin{table}[htbp]\\centering\n"
    "\\caption{MAE (THB) of the selected exponent under three independent seeds.}\n"
    "\\label{tab:seeds}\n\\small\n"
    "\\begin{tabular}{lccccc}\n\\toprule\n"
    "Seed & $t{+}20$ & $t{+}60$ & $t{+}120$ & $t{+}250$ & Overall \\\\\n\\midrule\n"
    + "\n".join(seed_rows)
    + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
)

# Scientific detail moved from the review manuscript to meet the journal's
# preferred main-paper length.
group_order = [
    "Meat & Poultry", "Aquatic & Marine", "Fresh Vegetables",
    "Organic Veg & Fruits", "Fruits", "Food Crops", "Oil Crops & Oils",
    "Wholesale Rice & Bags", "Wholesale Rice to Retailers", "Retail Rice",
    "Field Crops", "Feed & Raw Materials", "Miscellaneous / Unknown",
]
group_counts = eda.groupby(["group", "status"]).size().unstack(fill_value=0)
inventory_rows = []
for group in group_order:
    active_n = int(group_counts.loc[group, "active"])
    empty_n = int(group_counts.loc[group, "empty"])
    group_tex = group.replace("&", r"\&")
    inventory_rows.append(
        f"{group_tex} & {active_n} & {empty_n} & {active_n + empty_n} \\\\"
    )
inventory_rows.append(
    f"\\textbf{{Total}} & \\textbf{{{int((eda.status == 'active').sum())}}} & "
    f"\\textbf{{{int((eda.status == 'empty').sum())}}} & "
    f"\\textbf{{{len(eda)}}} \\\\"
)
active_eda = eda[eda["status"] == "active"]
stationarity_rows = []
for category, sub in active_eda.groupby("category", sort=True):
    stationarity_rows.append(
        f"{category} & {len(sub)} & "
        f"{100 * (sub.adf_p_min < 0.05).mean():.1f}\\% & "
        f"{100 * (sub.adf_p_max < 0.05).mean():.1f}\\% \\\\"
    )

tex.append(
    "\\section{Data cleaning and exploratory diagnostics}\n"
    "Zero prices are treated as missing, rolling 30-day IQR outliers are "
    "removed, remaining gaps are forward-filled after business-day alignment, "
    "and a one-percent repricing threshold defines the 404-product modeling "
    "universe. Table~\\ref{tab:inventory} reports the raw inventory; "
    "Table~\\ref{tab:stationarity} reports level-stationarity rates among the "
    "635 active products. Figures~\\ref{fig:acf}--\\ref{fig:corr} provide the "
    "full diagnostics summarized in Section~3 of the main paper.\n\n"
    "\\begin{table}[htbp]\\centering\n"
    "\\caption{Product counts and status by group.}\\label{tab:inventory}\\small\n"
    "\\begin{tabular}{lrrr}\\toprule\n"
    "Group & Active & Empty/all-zero & Total \\\\\n\\midrule\n"
    + "\n".join(inventory_rows)
    + "\n\\bottomrule\\end{tabular}\\end{table}\n\n"
    "\\begin{table}[htbp]\\centering\n"
    "\\caption{ADF level-stationarity profiles by market category "
    "($p<0.05$).}\\label{tab:stationarity}\\small\n"
    "\\begin{tabular}{lrrr}\\toprule\n"
    "Category & Active & Minimum price & Maximum price \\\\\n\\midrule\n"
    + "\n".join(stationarity_rows)
    + "\n\\bottomrule\\end{tabular}\\end{table}\n\n"
    "\\begin{figure}[htbp]\\centering\\includegraphics[width=\\linewidth]{acf_pacf.png}\n"
    "\\caption{Mean ACF and PACF in levels by product group.}\\label{fig:acf}\\end{figure}\n"
    "\\begin{figure}[htbp]\\centering\\includegraphics[width=\\linewidth]{price_trend.png}\n"
    "\\caption{Indexed monthly price trajectories of the six largest groups.}\\label{fig:trend}\\end{figure}\n"
    "\\begin{figure}[htbp]\\centering\\includegraphics[width=\\linewidth]{stationarity_by_group.png}\n"
    "\\caption{ADF stationarity rates by product group.}\\label{fig:adf}\\end{figure}\n"
    "\\begin{figure}[htbp]\\centering\\includegraphics[width=\\linewidth]{correlation_heatmap.png}\n"
    "\\caption{Price-level correlation among the 40 most liquid series; this "
    "deliberately integrated subset is not representative of the full universe, "
    "whose mean pairwise level correlation is 0.180.}\\label{fig:corr}\\end{figure}\n"
)

tex.append(r"""\section{Detailed model and loss specification}
All in-domain learned references use leakage-safe price lags, differences,
returns, rolling statistics, repricing counters, calendar coordinates, and
entity identifiers. LightGBM is direct by horizon; the neural references use a
joint multi-output head; ARIMA is local; and Chronos-Bolt-Base is zero-shot.
The released source contains the exact feature list and configurations.

The TFT maps product, category, and group identifiers to static embeddings.
Gated residual networks (GRNs) create context vectors for variable selection,
LSTM initialization, and static enrichment. With an optional projected residual
$\mathbf W_s\mathbf a$, a GRN is
\[
\operatorname{GRN}(\mathbf a,\mathbf c)=
\operatorname{LayerNorm}\{\mathbf W_s\mathbf a+
\operatorname{GLU}(\mathbf W_1\operatorname{ELU}
(\mathbf W_2\mathbf a+\mathbf W_3\mathbf c+\mathbf b_2)+\mathbf b_1)\}.
\]
The variable-selection network combines independently transformed variables as
\[
\tilde{\mathbf x}_t=\sum_{j=1}^{M}v_t^{(j)}
\operatorname{GRN}_{j}(\mathbf x_t^{(j)}),\qquad
\mathbf v_t=\operatorname{Softmax}(\operatorname{GRN}_v(\mathbf g_t,\mathbf c)).
\]
Interpretable multi-head attention uses head-specific query/key projections,
a shared value projection, and an averaged attention matrix,
\[
\tilde{\mathbf A}=\frac1M\sum_{m=1}^{M}
\operatorname{Softmax}\!\left(
\frac{(\mathbf Q\mathbf W_Q^{(m)})
(\mathbf K\mathbf W_K^{(m)})^\top}{\sqrt{d_{\rm attn}}}\right),
\quad
\operatorname{IMHA}=\tilde{\mathbf A}\mathbf V\mathbf W_V\mathbf W_O.
\]
The output heads target quantiles $\{0.1,0.5,0.9\}$ with pinball loss
$\rho_q(e)=e(q-\mathbb 1\{e<0\})$. The proposed objective is
\[
\mathcal L=\sum_{h=1}^{250}h^{-\gamma}
\sum_{q\in\{0.1,0.5,0.9\}}\rho_q(y_{t+h}-\hat y_{t+h,q}).
\]
Pinball-loss subgradients are bounded, so the paper interprets this as explicit
loss allocation across decoder steps rather than as a direct measurement of
gradient-vector conflict.

\begin{figure}[htbp]\centering
\includegraphics[width=.9\linewidth]{loss_weighting_curves.png}
\caption{Power-law loss schedules over the 250-step decoder.}
\label{fig:weights}
\end{figure}
""")

gamma_seed42 = gamma_sweep[
    (gamma_sweep["Seed"] == 42)
    & gamma_sweep["Horizon"].isin([20, 60, 120, 250])
]
gamma_rows = []
for gamma in [i / 2 for i in range(17)]:
    sub = gamma_seed42[gamma_seed42["Gamma"] == gamma].set_index("Horizon")
    cells = [
        f"{sub.loc[h, 'MAE']:.2f}/{sub.loc[h, 'SMAPE']:.2f}\\%"
        for h in [20, 60, 120, 250]
    ]
    gamma_rows.append(
        f"{gamma:.1f} & " + " & ".join(cells)
        + f" & {sub.MAE.mean():.2f} \\\\"
    )
tex.append(
    "\\section{Complete loss-exponent sweep}\n"
    "Table~\\ref{tab:gammafull} reports the complete seed-42 grid used for "
    "validation selection; Figure~\\ref{fig:gamma} shows the corresponding "
    "test-MAE curves. Each exponent is one run, so local non-monotonicity "
    "should not be interpreted as a smooth dose response.\n\n"
    "\\begin{table}[htbp]\\centering\\scriptsize\n"
    "\\caption{Seed-42 horizon-weighted TFT results (MAE THB/SMAPE).}"
    "\\label{tab:gammafull}\n"
    "\\resizebox{\\textwidth}{!}{\\begin{tabular}{cccccc}\\toprule\n"
    "$\\gamma$ & $t{+}20$ & $t{+}60$ & $t{+}120$ & $t{+}250$ & Overall MAE \\\\\n\\midrule\n"
    + "\n".join(gamma_rows)
    + "\n\\bottomrule\\end{tabular}}\\end{table}\n"
    "\\begin{figure}[htbp]\\centering\\includegraphics[width=.9\\linewidth]{mae_vs_gamma.png}\n"
    "\\caption{Horizon-specific and overall test MAE across the exponent grid.}"
    "\\label{fig:gamma}\\end{figure}\n"
)

tex.append(r"""\clearpage
\section{Qualitative forecasts and model diagnostics}
The first trajectory panel deliberately includes contrasting cases, while the
second uses one median-representative product per group. These single origins
are diagnostic examples, not substitutes for aggregate evaluation.
\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{qualitative_predictions.png}
\caption{Contrasting held-out trajectories from the earliest 2024 origin.}
\label{fig:qual}
\end{figure}
\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{typical_qualitative_predictions.png}
\caption{One median-representative held-out trajectory per commodity group.}
\label{fig:typical}
\end{figure}

Attention and variable-selection weights describe this fitted network, not
causal economic effects. The oldest encoder position receives the largest
averaged attention weight; observed price dominates historical selection;
group and product identifiers dominate the informative static covariates; and
day of week dominates known-future selection. Encoder length is constant in
this experiment, so its static-selection weight is an architectural artifact.
\begin{figure}[htbp]\centering
\includegraphics[width=.82\linewidth]{tft_attention_ablation.png}
\caption{Averaged temporal attention over the 30-step encoder.}
\label{fig:attn}
\end{figure}
\begin{figure}[htbp]\centering
\includegraphics[width=.82\linewidth]{tft_static_vars_ablation.png}
\caption{Static variable-selection weights.}
\label{fig:static}
\end{figure}
\begin{figure}[htbp]\centering
\includegraphics[width=.82\linewidth]{tft_encoder_vars_ablation.png}
\caption{Historical encoder variable-selection weights.}
\label{fig:encoder}
\end{figure}
\begin{figure}[htbp]\centering
\includegraphics[width=.82\linewidth]{tft_decoder_vars_ablation.png}
\caption{Known-future decoder variable-selection weights.}
\label{fig:decoder}
\end{figure}

\clearpage
\section{Target-normalizer level-shift diagnostic}
For 401 products, define the raw-price proxy
\[
\operatorname{gap}_p=100\,
\frac{y^{2023}_{p,\mathrm{last}}-\bar y^{2018:2022}_p}
{\bar y^{2018:2022}_p}.
\]
This proxy is not the transformed internal normalizer center. Its absolute
value exceeds 10\% for 283 products (70.6\%), with mean 30.4\%, median 18.4\%,
and maximum 332\%. Absolute gap correlates with relative selected-model error
at $t{+}20$ (Spearman $r=0.146$, unadjusted $p=0.007$), reverses sign at
$t{+}120$ ($r=-0.206$, $p<0.001$) and $t{+}250$ ($r=-0.188$, $p<0.001$), and
does not identify a causal normalizer effect. The per-product inputs are
released in \texttt{normalizer\_staleness\_check.csv} and
\texttt{gap\_vs\_performance.csv}.
""")

tex.append(r"""\section{ARIMA order}
The local statistical reference uses $\text{ARIMA}(1,1,0)$, chosen a priori from the lag structure documented in Section~3.3 of the main paper (a unit root with a single partial-autocorrelation spike at lag one). We did not run a per-product order search for two reasons. First, Table~5 of the main paper shows the fitted $\text{ARIMA}(1,1,0)$ is already numerically indistinguishable from the persistence baseline at every horizon (differences of at most 0.01 THB), and the closed-form forecast function in Section~4.3 of the main paper explains why any low-order specification on a near-unit-root series must collapse onto that baseline. Second, the persistence baseline itself is retained as a separate reference, so the comparison against the strongest local statistical rule is already made directly.

\section{Summary}
%%SUMMARY%%
\end{document}
""")

out_tex = "\n".join(tex)

# summary paragraph written from the actual outcome; tuning effect is measured
# against the same-pipeline rerun of the original Table 5 configuration
lines = []
for fam in ["lightgbm", "mlp", "lstm", "transformer"]:
    sel = final[(final.family == fam) & (final.which == "selected")].iloc[0]
    pub = final[(final.family == fam) & (final.which == "published")].iloc[0]
    delta = 100 * (sel.test_mae_mean - pub.test_mae_mean) / pub.test_mae_mean
    lines.append((fam, sel, pub.test_mae_mean, delta))

any_beats_baseline = any(sel.test_mae_mean < baseline_test for _, sel, _, _ in lines)
any_beats_tft = any(sel.test_mae_mean < 39.76 for _, sel, _, _ in lines)
frag = []
for fam, sel, pub_overall, delta in lines:
    frag.append(f"{FAMILY_NAMES[fam]} moves from {pub_overall:.2f} (original configuration, rerun) to {sel.test_mae_mean:.2f} THB ({delta:+.1f}\\%)")
summary = ("Under a matched 10-configuration validation-based architecture-search budget per class, "
           + "; ".join(frag) + ". ")
if not any_beats_tft:
    summary += ("No validation-selected reference configuration approaches the selected TFT of the main paper "
                "(39.76 THB overall), and ")
    if not any_beats_baseline:
        summary += "none outperforms the matched persistence baseline overall. "
    else:
        summary += "the ranking of model classes in Table~5 of the main paper is unchanged. "
    summary += ("The conclusions of the main paper are therefore insensitive to reference-model tuning "
                "at this architecture-search budget, and the configurations reported in Table~5 give each reference class "
                "a representative rather than handicapped operating point.")
else:
    summary += ("A validation-selected reference configuration outperforms the selected TFT of the main paper; "
                "this materially changes the comparison and is disclosed prominently in the revised main text.")

out_tex = out_tex.replace("%%SUMMARY%%", summary)

path = os.path.join(ROOT, "paper", "supplementary.tex")
open(path, "w", encoding="utf-8", newline="\n").write(out_tex)
print("wrote", path)
print("\nSUMMARY PARAGRAPH:\n" + summary)
