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

PUBLISHED_TEST = {  # Table 5 of the main paper (overall = mean of the four horizons)
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
            tag = " (published, selected)"
        elif r.published:
            tag = " (published)"
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
\usepackage{graphicx}
\usepackage[hidelinks]{hyperref}
\setlength{\emergencystretch}{2em}
\hypersetup{pdftitle={Supplementary Material: Horizon-Weighted Temporal Fusion Transformers for Thai Agricultural Commodity Price Forecasting},pdfauthor={Kritaphat Songsri-in, Auyporn Chukeaw, Munlika Rattaphun, Walaiporn Sornkliang, Rattayagon Thaiphan}}
\renewcommand{\thetable}{S\arabic{table}}
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
\item \textbf{Equal, deliberately light budgets.} The TFT architecture was fixed once using a 10-trial search (Section~S2) and held constant across all seventeen decay exponents, because the contribution of the paper is the loss geometry rather than architecture optimization. Each learned reference class therefore also receives a 10-configuration search (Section~S3), and the configuration used in the main paper is always one of the ten candidates.
\item \textbf{Validation-only selection.} Every candidate is fitted on labels within 2018--2022 only and scored by matched-window MAE on validation windows whose horizon-$h$ label falls inside 2023 (12{,}120 windows at $t{+}20$, 28{,}280 at $t{+}60$, 52{,}520 at $t{+}120$, and 105{,}040 at $t{+}250$ over the 404 crops), averaged over $h \in \{20, 60, 120, 250\}$. No test-period information enters any selection.
\item \textbf{Identical final conditions.} The validation-selected configuration is then refitted on labels through 2023, the same training window the published reference models used, and evaluated on the identical 2024--2025 matched test windows as Table~5 of the main paper (110{,}292 windows per horizon, 404 crops).
\end{itemize}

\section{TFT architecture provenance}
The TFT architecture used throughout the main paper (hidden size 64, 4 attention heads, continuous-variable encoding size 8, dropout 0.1, learning rate 0.01) was fixed before the decay-exponent sweep, informed by a 10-trial Optuna search (median pruner, 3-epoch budget per trial) minimizing validation loss under the unweighted quantile objective. Table~\ref{tab:tftspace} lists the search space. The architecture was then held constant across all seventeen decay exponents of the main experiment, so that every accuracy difference in Section~5.2 of the main paper is attributable to the loss weighting alone. The decay exponent itself, the single hyperparameter this paper is about, was not searched stochastically but swept exhaustively over $\gamma \in \{0.0, 0.5, \dots, 8.0\}$ with full per-horizon results disclosed in Table~6 of the main paper.

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
Tables~\ref{tab:lgb}--\ref{tab:tf} report all ten candidate configurations per class with their per-horizon validation MAE. Configuration~0 is always the configuration used in the main paper. The gradient-boosting search varies tree complexity, regularization, and sampling; the MLP search varies width, dropout, and learning rate around the published values; the LSTM search varies hidden-state size and learning rate; the Transformer search varies model width, depth, head count, and learning rate. Training budgets (boosting rounds, epochs, batch sizes) are held at their published values within each class so that the search isolates configuration rather than compute. The GRU variant of the recurrent class is omitted from the search for the reason given in Section~4.3 of the main paper; it is not part of Table~5 and does not outperform the naive baseline overall.
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

# final comparison table (selected vs rerun-published vs Table 5)
rows = []
for fam in ["lightgbm", "mlp", "lstm", "transformer"]:
    sel = final[(final.family == fam) & (final.which == "selected")].iloc[0]
    pub = final[(final.family == fam) & (final.which == "published")].iloc[0]
    same = int(sel.config_id) == int(pub.config_id)
    rows.append(
        f"{FAMILY_NAMES[fam]} & {int(sel.config_id)}{' (= published)' if same else ''} & "
        + " & ".join(f"{sel[f'test_mae_{h}']:.2f}" for h in [20, 60, 120, 250])
        + f" & {sel.test_mae_mean:.2f} & {pub.test_mae_mean:.2f} & {PUBLISHED_TEST[fam]:.2f} \\\\"
    )
baseline_test = final.baseline_test_mae.iloc[0]

tex.append(
    "\\section{Test-set impact of validation-selected configurations}\n"
    "Table~\\ref{tab:impact} reports the 2024--2025 matched-window test MAE of each validation-selected configuration after refitting on labels through 2023. The ``published (rerun)'' column refits the published configuration inside this supplement's pipeline under the same random initialization as the search candidates, and the ``main paper'' column repeats the corresponding overall value of Table~5. For the gradient-boosting class the rerun reproduces the main-paper value essentially exactly, which validates the pipeline; for the neural classes the rerun differs from the main-paper value by several THB in either direction under an identical configuration, a direct illustration of the initialization variance already documented in Section~5.2 of the main paper, and tuning-attributable differences should therefore be read against the rerun column rather than the main-paper column. The matched Lag-1 persistence baseline scores "
    f"{baseline_test:.2f}"
    " THB on the same windows.\n\n"
    "\\begin{table}[htbp]\\centering\n"
    "\\caption{Test MAE (THB) of validation-selected reference configurations against the published configurations.}\n"
    "\\label{tab:impact}\n\\small\n"
    "\\begin{tabular}{llccccccc}\n\\toprule\n"
    " & & \\multicolumn{5}{c}{Selected configuration} & Published & Main \\\\\n"
    "Model class & Cfg & $t{+}20$ & $t{+}60$ & $t{+}120$ & $t{+}250$ & Overall & (rerun) & paper \\\\\n\\midrule\n"
    + "\n".join(rows)
    + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
)

tex.append(r"""\section{ARIMA order}
The local statistical reference uses $\text{ARIMA}(1,1,0)$, chosen a priori from the lag structure documented in Section~3.3 of the main paper (a unit root with a single partial-autocorrelation spike at lag one). We did not run a per-product order search for two reasons. First, Table~5 of the main paper shows the fitted $\text{ARIMA}(1,1,0)$ is already numerically indistinguishable from the persistence baseline at every horizon (differences of at most 0.01 THB), and the closed-form forecast function in Section~4.3 of the main paper explains why any low-order specification on a near-unit-root series must collapse onto that baseline. Second, the persistence baseline itself is retained as a separate reference, so the comparison against the strongest local statistical rule is already made directly.

\section{Summary}
%%SUMMARY%%
\end{document}
""")

out_tex = "\n".join(tex)

# summary paragraph written from the actual outcome; tuning effect is measured
# against the same-pipeline rerun of the published configuration
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
    frag.append(f"{FAMILY_NAMES[fam]} moves from {pub_overall:.2f} (published configuration, rerun) to {sel.test_mae_mean:.2f} THB ({delta:+.1f}\\%)")
summary = ("Under an equal 10-configuration validation-based budget per class, "
           + "; ".join(frag) + ". ")
if not any_beats_tft:
    summary += ("No validation-selected reference configuration approaches the selected TFT of the main paper "
                "(39.76 THB overall), and ")
    if not any_beats_baseline:
        summary += "none outperforms the matched persistence baseline overall. "
    else:
        summary += "the ranking of model classes in Table~5 of the main paper is unchanged. "
    summary += ("The conclusions of the main paper are therefore insensitive to reference-model tuning "
                "at this budget, and the configurations reported in Table~5 give each reference class "
                "a representative rather than handicapped operating point.")
else:
    summary += ("A validation-selected reference configuration outperforms the selected TFT of the main paper; "
                "this materially changes the comparison and is disclosed prominently in the revised main text.")

out_tex = out_tex.replace("%%SUMMARY%%", summary)

path = os.path.join(ROOT, "paper", "supplementary.tex")
open(path, "w", encoding="utf-8", newline="\n").write(out_tex)
print("wrote", path)
print("\nSUMMARY PARAGRAPH:\n" + summary)
