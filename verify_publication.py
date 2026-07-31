"""Fail-fast audit of the files submitted with the manuscript.

This complements ``verify_experiments.py``: it checks journal-facing limits,
submission placeholders, citation and figure integrity, the selected result
contract, and the implemented horizon weighting.  Run it after building PDFs.
"""
from __future__ import annotations

import json
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pypdf import PdfReader
from scipy import stats

from paths import CHAMPION_NPZ, PAPER_DIR, RESULTS_DIR, ROOT
from src.models.loss import HorizonWeightedQuantileLoss


MANUSCRIPT = PAPER_DIR / "final_paper.md"
EXPECTED_HORIZONS = [20, 60, 120, 250]
EXPECTED_MAE = np.array([26.5112204, 33.8053040, 45.7540072, 52.9746992])
EXPECTED_BASELINE_MAE = np.array([15.1096730, 30.8591872, 51.0263580, 83.8405343])
PLACEHOLDERS = (
    r"\[username\]",
    r"\[author list",
    r"\[complete before",
    r"\[corresponding author",
    r"\[or:\s*at the public",
    r"must be confirmed before submission",
    r"should be re-verified before submission",
    r"repository to be linked",
    r"reference details .* should be verified",
)


class Audit:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passes: list[str] = []
        self.warnings: list[str] = []

    def check(self, condition: bool, success: str, failure: str) -> None:
        if condition:
            self.passes.append(success)
        else:
            self.failures.append(failure)


def section(text: str, start: str, end: str) -> str:
    match = re.search(
        rf"^##\s+{re.escape(start)}\s*$\n(.*?)^\*\*{re.escape(end)}:\*\*",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if not match:
        raise ValueError(f"could not locate {start} section")
    return match.group(1).strip()


def word_count(text: str) -> int:
    clean = re.sub(r"[*`$\\{}]", " ", text)
    return len(re.findall(r"[A-Za-z0-9]+(?:[-–][A-Za-z0-9]+)*", clean))


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running_max = 0.0
    m = len(values)
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * values[idx])
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def markdown_table(text: str, number: int) -> tuple[list[str], list[list[str]]]:
    lines = text.splitlines()
    start = next(
        i for i, line in enumerate(lines)
        if line.startswith(f"#### Table {number}:")
    )
    raw: list[str] = []
    for line in lines[start + 1:]:
        if not line.startswith("|"):
            if raw:
                break
            continue
        raw.append(line)
    if len(raw) < 3:
        raise ValueError(f"Table {number} has no parseable rows")

    def cells(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    return cells(raw[0]), [cells(line) for line in raw[2:]]


def cell_numbers(cell: str) -> list[float]:
    clean = (
        cell.replace("**", "")
        .replace("$", "")
        .replace(",", "")
        .replace("−", "-")
        .replace("–", "-")
    )
    return [
        float(value)
        for value in re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", clean)
    ]


def audit_manuscript(audit: Audit, text: str) -> None:
    abstract = section(text, "Abstract", "Keywords")
    count = word_count(abstract)
    audit.check(count <= 250, f"abstract: {count}/250 words", f"abstract has {count} words (limit 250)")

    keyword_match = re.search(r"^\*\*Keywords:\*\*\s*(.+)$", text, re.MULTILINE)
    keywords = [k.strip() for k in keyword_match.group(1).split(";")] if keyword_match else []
    audit.check(1 <= len(keywords) <= 7, f"keywords: {len(keywords)}/7", f"found {len(keywords)} keywords (allowed 1-7)")

    manuscript_highlights = re.findall(
        r"^-\s+(.+)$",
        text[text.index("## Highlights") : text.index("## Abstract")],
        re.MULTILINE,
    )
    separate_highlights = re.findall(
        r"^-\s+(.+)$", (PAPER_DIR / "highlights.txt").read_text(encoding="utf-8"), re.MULTILINE
    )
    lengths = [len(item.strip()) for item in separate_highlights]
    audit.check(3 <= len(lengths) <= 5, f"highlights: {len(lengths)} items", f"highlights has {len(lengths)} items (allowed 3-5)")
    audit.check(all(n <= 85 for n in lengths), f"highlight lengths: {lengths}", f"a highlight exceeds 85 characters: {lengths}")
    audit.check(manuscript_highlights == separate_highlights, "separate highlights match manuscript", "paper/highlights.txt differs from manuscript highlights")

    images = re.findall(r"!\[[^]]*\]\((images/[^)]+)\)", text)
    missing = [name for name in images if not (PAPER_DIR / name).is_file()]
    audit.check(len(images) == 12, "manuscript embeds 12 figures", f"expected 12 figures, found {len(images)}")
    audit.check(not missing, "all figure files exist", f"missing figure files: {missing}")

    body, references = text.split("## References", maxsplit=1)
    body_without_math = re.sub(r"\$[^$]*\$", "", body)
    cited: set[int] = set()
    for citation in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", body_without_math):
        cited.update(int(item.strip()) for item in citation.split(","))
    listed = {int(item) for item in re.findall(r"^\[(\d+)\]", references, re.MULTILINE)}
    audit.check(cited <= listed, "every numeric citation has a reference", f"missing references for citations: {sorted(cited - listed)}")
    audit.check(listed <= cited, "every reference is cited", f"uncited references: {sorted(listed - cited)}")

    expected_contact = "kritaphat_son@nstru.ac.th"
    audit.check(expected_contact in text, "corresponding email is populated", "corresponding email is absent")
    audit.check("80280" in text and "1 Moo 4" in text, "full postal address is populated", "full affiliation postal address is absent")
    audit.check(
        "does not grow with the absolute residual" in text
        and "measure gradient-vector conflict directly" in text,
        "pinball-loss mechanism is stated without false gradient-magnitude inference",
        "manuscript lacks the bounded-subgradient qualification",
    )
    audit.check(
        "value projection" in text
        and "head-specific" in text
        and "averaged matrix" in text,
        "TFT interpretable-attention sharing is described correctly",
        "TFT attention sharing/averaging description is incomplete",
    )
    audit.check(
        "This is not a formal split-conformal construction" in text
        and "without a finite-sample guarantee" in text,
        "interval scaling is explicitly separated from formal conformal validity",
        "interval-validity limitation is missing",
    )
    audit.check(
        "| Global LightGBM | 29.37 | 43.37 | 49.69 | 56.88 | 44.83 |" in text
        and "| **Horizon-Weighted Quantile TFT (ours)** | 26.51 | 33.81 | **45.75** | **52.97** | **39.76** |" in text,
        "Table 5 bolds the Horizon-Weighted Quantile TFT, not LightGBM, at t+250",
        "Table 5 has inconsistent emphasis in the t+250 column",
    )
    audit.check(
        "our own unweighted TFT" not in text
        and "Unweighted TFT ($\\gamma=0$, ours)" not in text
        and "| TFT | 69.74 | 76.37 | 73.94 | 71.37 | 72.85 |" in text,
        "Table 5 labels the standard baseline simply as TFT",
        "Table 5 does not use the requested TFT baseline label",
    )
    forbidden_overclaims = (
        "resolves multi-scale gradient conflict",
        "far-future errors dominate joint training",
        "conformally calibrated intervals",
    )
    hits = [phrase for phrase in forbidden_overclaims if phrase.lower() in text.lower()]
    audit.check(not hits, "known methodological overclaims are absent", f"overclaim phrases remain: {hits}")


def audit_placeholders(audit: Audit) -> None:
    targets = [
        PAPER_DIR / "final_paper.md",
        PAPER_DIR / "cover_letter.md",
        PAPER_DIR / "main.tex",
        PAPER_DIR / "supplementary.tex",
        PAPER_DIR / "cover_letter.tex",
    ]
    hits: list[str] = []
    for path in targets:
        if not path.exists():
            hits.append(f"missing {path.relative_to(ROOT)}")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in PLACEHOLDERS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                hits.append(f"{path.relative_to(ROOT)} matches /{pattern}/")
    audit.check(not hits, "no submission placeholders in deliverables", "; ".join(hits))


def audit_results(audit: Audit) -> None:
    metrics_path = RESULTS_DIR / "champion_gamma45_metrics.csv"
    selection_path = RESULTS_DIR / "gamma_validation_summary.csv"
    provenance_path = RESULTS_DIR / "publication_provenance.json"
    dm_path = RESULTS_DIR / "dm_tests.csv"
    multiseed_path = RESULTS_DIR / "champion_gamma45_multiseed_metrics.csv"
    audit.check(metrics_path.is_file(), "champion metric table exists", "missing champion metric table")
    audit.check(selection_path.is_file(), "validation selection table exists", "missing validation selection table")
    audit.check(provenance_path.is_file(), "publication provenance exists", "missing publication provenance")
    audit.check(dm_path.is_file(), "product-level test table exists", "missing product-level test table")
    audit.check(multiseed_path.is_file(), "multi-seed metric table exists", "missing multi-seed metric table")
    if not all(path.is_file() for path in (metrics_path, selection_path, provenance_path, dm_path, multiseed_path)):
        return

    metrics = pd.read_csv(metrics_path).sort_values("Horizon")
    audit.check(metrics.Horizon.tolist() == EXPECTED_HORIZONS, "metric horizons are exact", f"metric horizons are {metrics.Horizon.tolist()}")
    audit.check(np.allclose(metrics.MAE, EXPECTED_MAE, atol=1e-6), "reported TFT MAEs match audit contract", f"TFT MAEs changed: {metrics.MAE.tolist()}")
    audit.check(np.allclose(metrics.Baseline_MAE, EXPECTED_BASELINE_MAE, atol=1e-6), "reported baseline MAEs match audit contract", f"baseline MAEs changed: {metrics.Baseline_MAE.tolist()}")
    audit.check((metrics.CalPICoverage.between(0.75, 0.91)).all(), "calibrated coverage is within reported range", f"calibrated coverage changed: {metrics.CalPICoverage.tolist()}")

    selection = pd.read_csv(selection_path).sort_values("Gamma")
    expected_gammas = [i / 2 for i in range(17)]
    selected = float(selection.loc[selection.MAE_Overall.idxmin(), "Gamma"])
    audit.check(selection.Gamma.tolist() == expected_gammas, "validation audit covers all 17 gamma values", f"validation gamma values are {selection.Gamma.tolist()}")
    audit.check(selected == 4.5, "held-out validation selects gamma=4.5", f"held-out validation selects gamma={selected}")
    audit.check(
        selection[[f"N_{h}" for h in EXPECTED_HORIZONS]].iloc[0].tolist()
        == [12120, 28280, 52520, 105040],
        "validation sample counts are horizon-specific and exact",
        "validation sample counts differ from the evaluation protocol",
    )

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    audit.check(provenance.get("selected_gamma") == 4.5, "provenance records gamma=4.5", "provenance selected_gamma is incorrect")
    audit.check(
        provenance.get("sweep_seed") == 42
        and provenance.get("selected_artifact_seed") == 42
        and provenance.get("seed_sensitivity") == [42, 43, 44],
        "provenance distinguishes selection seed from sensitivity seeds",
        "provenance seed fields are incomplete",
    )
    audit.check(
        "no formal coverage guarantee" in provenance.get("interval_method", ""),
        "provenance qualifies empirical interval scaling",
        "provenance overstates interval validity",
    )

    dm = pd.read_csv(dm_path)
    horizon_dm = dm[dm["horizon"].isin(EXPECTED_HORIZONS)].copy()
    expected_t_holm = holm_adjust(horizon_dm["p_ttest"].to_numpy())
    expected_w_holm = holm_adjust(horizon_dm["p_wilcoxon"].to_numpy())
    audit.check(
        {"p_ttest_holm", "p_wilcoxon_holm"} <= set(dm.columns),
        "Holm-adjusted p-value columns exist",
        "Holm-adjusted p-value columns are missing",
    )
    if {"p_ttest_holm", "p_wilcoxon_holm"} <= set(dm.columns):
        audit.check(
            np.allclose(horizon_dm["p_ttest_holm"], expected_t_holm)
            and np.allclose(horizon_dm["p_wilcoxon_holm"], expected_w_holm),
            "Holm-adjusted p-values reproduce exactly",
            "stored Holm-adjusted p-values do not reproduce",
        )
        significant = horizon_dm.set_index("horizon")
        audit.check(
            all(
                significant.loc[h, "p_ttest_holm"] < 0.01
                and significant.loc[h, "p_wilcoxon_holm"] < 0.01
                for h in (20, 120, 250)
            )
            and significant.loc[60, "p_ttest_holm"] > 0.05
            and significant.loc[60, "p_wilcoxon_holm"] > 0.05,
            "Holm conclusions match the manuscript",
            "Holm significance pattern changed",
        )

    multiseed = pd.read_csv(multiseed_path)
    multiseed = multiseed[
        multiseed["Seed"].isin([42, 43, 44])
        & multiseed["Horizon"].isin(EXPECTED_HORIZONS)
    ].copy()
    complete = (
        multiseed.groupby("Seed")["Horizon"].apply(lambda s: sorted(s.tolist())).to_dict()
        == {42: EXPECTED_HORIZONS, 43: EXPECTED_HORIZONS, 44: EXPECTED_HORIZONS}
    )
    audit.check(complete, "three-seed sensitivity grid is complete", "three-seed sensitivity grid is incomplete")
    if complete:
        overall = multiseed.groupby("Seed")["MAE"].mean()
        baseline = multiseed.groupby("Horizon")["Baseline_MAE"].first()
        long_horizon_wins = all(
            row.MAE < row.Baseline_MAE
            for row in multiseed.itertuples()
            if row.Horizon in (120, 250)
        )
        audit.check(
            np.all(overall.to_numpy() < baseline.mean()) and long_horizon_wins,
            "all three seeds beat persistence overall and at long horizons",
            "multi-seed robustness claim no longer holds",
        )
        audit.check(
            np.allclose(overall.to_numpy(), [39.7613076859, 42.2175683101, 43.7049012957], atol=1e-6),
            "three-seed overall MAEs match the manuscript",
            f"three-seed overall MAEs changed: {overall.tolist()}",
        )

    if CHAMPION_NPZ.exists():
        with np.load(CHAMPION_NPZ) as z:
            recomputed = [
                float(np.mean(np.abs(z[f"test_h{h}_actual"] - z[f"test_h{h}_median"])))
                for h in EXPECTED_HORIZONS
            ]
        audit.check(np.allclose(recomputed, metrics.MAE, atol=1e-12), "CSV metrics reproduce from champion NPZ", "CSV metrics do not reproduce from champion NPZ")
        audit.check(
            sha256(CHAMPION_NPZ) == provenance.get("artifact_sha256"),
            "champion artifact hash matches provenance",
            "champion artifact hash differs from provenance",
        )
    else:
        audit.passes.append("champion NPZ not local; aggregate/provenance audit used")


def audit_reported_tables(audit: Audit, text: str) -> None:
    """Cross-check manuscript Tables 5--9 against their versioned result sources."""

    def label(cell: str) -> str:
        return re.sub(r"[*$`]", "", cell).replace("\\", "").strip()

    def find_row(rows: list[list[str]], prefix: str) -> list[str]:
        for row in rows:
            if label(row[0]).startswith(prefix):
                return row
        raise ValueError(f"missing table row starting with {prefix!r}")

    try:
        # Table 5
        _, rows5 = markdown_table(text, 5)
        base = pd.read_csv(RESULTS_DIR / "metrics_by_horizon.csv")
        extra = pd.read_csv(RESULTS_DIR / "extra_baselines.csv").set_index("model")
        chronos = pd.read_csv(RESULTS_DIR / "chronos_zeroshot.csv").iloc[0]
        sweep = pd.read_csv(RESULTS_DIR / "gamma_sweep_summary.csv")
        champion = pd.read_csv(RESULTS_DIR / "champion_gamma45_metrics.csv").set_index("Horizon")

        def metric_row(paradigm: str, model: str) -> list[float]:
            sub = base[
                (base["paradigm"] == paradigm)
                & (base["model_name"] == model)
                & base["horizon"].isin(EXPECTED_HORIZONS)
            ]
            by_h = sub.groupby("horizon")["MAE"].mean()
            vals = [float(by_h.loc[h]) for h in EXPECTED_HORIZONS]
            return vals + [float(np.mean(vals))]

        expected5: dict[str, list[float]] = {
            "Lag-1 Persistence": metric_row("local", "baseline"),
            "Seasonal-Naive": [float(extra.loc["snaive", f"mae_{h}"]) for h in EXPECTED_HORIZONS]
            + [float(extra.loc["snaive", "mae_overall"])],
            "Drift": [float(extra.loc["drift", f"mae_{h}"]) for h in EXPECTED_HORIZONS]
            + [float(extra.loc["drift", "mae_overall"])],
            "ARIMA": metric_row("local", "arima"),
            "Global LightGBM": metric_row("global", "lightgbm"),
            "Global MLP": metric_row("global", "mlp"),
            "Global LSTM": metric_row("global", "lstm"),
            "Global Transformer": metric_row("global", "transformer"),
            "Chronos-Bolt": [float(chronos[f"mae_{h}"]) for h in EXPECTED_HORIZONS]
            + [float(chronos["mae_overall"])],
        }
        gamma0 = sweep[
            (sweep["Gamma"] == 0)
            & (sweep["Seed"] == 42)
            & sweep["Horizon"].isin(EXPECTED_HORIZONS)
        ].set_index("Horizon")
        expected5["TFT"] = [float(gamma0.loc[h, "MAE"]) for h in EXPECTED_HORIZONS]
        expected5["TFT"].append(float(np.mean(expected5["TFT"])))
        expected5["Horizon-Weighted Quantile TFT"] = [float(champion.loc[h, "MAE"]) for h in EXPECTED_HORIZONS]
        expected5["Horizon-Weighted Quantile TFT"].append(
            float(np.mean(expected5["Horizon-Weighted Quantile TFT"]))
        )

        for prefix, expected in expected5.items():
            row = find_row(rows5, prefix)
            actual = [cell_numbers(cell)[0] for cell in row[1:6]]
            if not np.allclose(actual, expected, atol=0.011):
                raise ValueError(f"Table 5 {prefix}: {actual} != {expected}")
        audit.passes.append("Table 5 reproduces from reference result files")

        # Table 6
        _, rows6 = markdown_table(text, 6)
        seed42 = sweep[
            (sweep["Seed"] == 42)
            & sweep["Horizon"].isin(EXPECTED_HORIZONS)
        ]
        for gamma in [i / 2 for i in range(17)]:
            row = next(row for row in rows6 if cell_numbers(row[0]) == [gamma])
            sub = seed42[seed42["Gamma"] == gamma].set_index("Horizon")
            for idx, h in enumerate(EXPECTED_HORIZONS, start=1):
                actual = cell_numbers(row[idx])
                expected = [float(sub.loc[h, "MAE"]), float(sub.loc[h, "SMAPE"])]
                if len(actual) < 2 or not np.allclose(actual[:2], expected, atol=0.011):
                    raise ValueError(f"Table 6 gamma={gamma}, h={h}: {actual} != {expected}")
            overall = cell_numbers(row[5])[0]
            if not np.isclose(overall, sub["MAE"].mean(), atol=0.011):
                raise ValueError(f"Table 6 gamma={gamma} overall mismatch")
        audit.passes.append("Table 6 reproduces from the seed-42 sweep")

        # Table 7
        _, rows7 = markdown_table(text, 7)
        for h in EXPECTED_HORIZONS:
            row = next(row for row in rows7 if cell_numbers(row[0]) == [float(h)])
            expected_base = [
                float(champion.loc[h, "Baseline_MAE"]),
                float(champion.loc[h, "Baseline_SMAPE"]),
                float(champion.loc[h, "Baseline_RMSE"]),
            ]
            expected_model = [
                float(champion.loc[h, "MAE"]),
                float(champion.loc[h, "SMAPE"]),
                float(champion.loc[h, "RMSE"]),
            ]
            if not np.allclose(cell_numbers(row[1])[:3], expected_base, atol=0.011):
                raise ValueError(f"Table 7 baseline h={h} mismatch")
            if not np.allclose(cell_numbers(row[2])[:3], expected_model, atol=0.011):
                raise ValueError(f"Table 7 selected model h={h} mismatch")
        overall_row = find_row(rows7, "Overall")
        expected_base_overall = champion[
            ["Baseline_MAE", "Baseline_SMAPE", "Baseline_RMSE"]
        ].mean().to_numpy()
        expected_model_overall = champion[["MAE", "SMAPE", "RMSE"]].mean().to_numpy()
        if not np.allclose(cell_numbers(overall_row[1])[:3], expected_base_overall, atol=0.011):
            raise ValueError("Table 7 overall baseline mismatch")
        if not np.allclose(cell_numbers(overall_row[2])[:3], expected_model_overall, atol=0.011):
            raise ValueError("Table 7 overall model mismatch")
        audit.passes.append("Table 7 point metrics reproduce from selected-model results")

        # Table 8
        _, rows8 = markdown_table(text, 8)
        for h in EXPECTED_HORIZONS:
            row = next(row for row in rows8 if cell_numbers(row[0]) == [float(h)])
            actual = [cell_numbers(cell)[0] for cell in (row[1], row[2], row[4], row[5])]
            expected = [
                float(champion.loc[h, "Baseline_MAE_2024"]),
                float(champion.loc[h, "MAE_2024"]),
                float(champion.loc[h, "Baseline_MAE_2025"]),
                float(champion.loc[h, "MAE_2025"]),
            ]
            if not np.allclose(actual, expected, atol=0.011):
                raise ValueError(f"Table 8 h={h}: {actual} != {expected}")
        audit.passes.append("Table 8 reproduces from year-split selected-model results")

        # Table 9
        _, rows9 = markdown_table(text, 9)
        for h in EXPECTED_HORIZONS:
            row = next(row for row in rows9 if cell_numbers(row[0]) == [float(h)])
            actual = [cell_numbers(cell)[0] for cell in row[1:4]]
            expected = [
                float(champion.loc[h, "CalPIScale"]),
                float(champion.loc[h, "CalPIWidth"]),
                100.0 * float(champion.loc[h, "CalPICoverage"]),
            ]
            if not np.allclose(actual, expected, atol=0.051):
                raise ValueError(f"Table 9 h={h}: {actual} != {expected}")
        audit.passes.append("Table 9 reproduces from interval-scaling results")
    except Exception as exc:
        audit.failures.append(f"manuscript table audit failed: {exc}")


def audit_secondary_claims(audit: Audit) -> None:
    """Verify the versioned per-product diagnostics used in limitations."""
    normalizer_path = RESULTS_DIR / "normalizer_staleness_check.csv"
    gap_path = RESULTS_DIR / "gap_vs_performance.csv"
    signal_path = RESULTS_DIR / "cross_product_signal_test.csv"
    eda_path = RESULTS_DIR / "eda_product_summary.csv"
    required = (normalizer_path, gap_path, signal_path, eda_path)
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        audit.failures.append(f"missing secondary-claim artifacts: {missing}")
        return

    normalizer = pd.read_csv(normalizer_path)
    recomputed_gap = 100.0 * (
        normalizer["last_price"] - normalizer["train_mean"]
    ) / normalizer["train_mean"]
    audit.check(
        np.allclose(recomputed_gap, normalizer["gap_pct"]),
        "normalizer level-shift formula reproduces",
        "normalizer level-shift formula does not reproduce",
    )
    abs_gap = normalizer["gap_pct"].abs()
    audit.check(
        len(normalizer) == 401
        and int((abs_gap > 10).sum()) == 283
        and np.isclose(abs_gap.mean(), 30.382491, atol=1e-6)
        and np.isclose(abs_gap.median(), 18.444757, atol=1e-6)
        and np.isclose(abs_gap.max(), 332.174661, atol=1e-6)
        and np.isclose((normalizer["gap_pct"] > 0).mean(), 0.7605985, atol=1e-6),
        "normalizer diagnostic summary reproduces",
        "normalizer diagnostic summary changed",
    )

    gap = pd.read_csv(gap_path)
    expected_correlations = {
        20: (0.1458839094, 0.0067200095),
        120: (-0.2060152472, 0.0000700982),
        250: (-0.1878709793, 0.0002956416),
    }
    correlations_ok = True
    for h, expected in expected_correlations.items():
        pair = gap[["abs_gap", f"rel{h}"]].dropna()
        result = stats.spearmanr(pair["abs_gap"], pair[f"rel{h}"])
        correlations_ok &= np.allclose([result.statistic, result.pvalue], expected, atol=1e-9)
    audit.check(
        correlations_ok,
        "normalizer-performance associations reproduce",
        "normalizer-performance associations changed",
    )

    signal = pd.read_csv(signal_path)
    audit.check(
        np.isclose(signal["corr"].mean(), 0.997749, atol=1e-6)
        and np.isclose(signal["improvement_pct"].mean(), 0.033688, atol=1e-6),
        "cross-product proxy summary reproduces",
        "cross-product proxy summary changed",
    )

    eda = pd.read_csv(eda_path)
    counts = eda["status"].value_counts().to_dict()
    audit.check(
        len(eda) == 729 and counts.get("active") == 635 and counts.get("empty") == 94,
        "EDA product-universe counts reproduce",
        f"EDA product-universe counts changed: {counts}",
    )


def audit_loss(audit: Audit) -> None:
    loss = HorizonWeightedQuantileLoss(gamma=2.0)
    target = torch.zeros((1, 3), dtype=torch.float32)
    prediction = torch.ones((1, 3, 3), dtype=torch.float32)
    per_horizon = loss.loss(prediction, target).mean(dim=(0, 2)).detach().cpu().numpy()
    ratios = per_horizon / per_horizon[0]
    audit.check(np.allclose(ratios, [1.0, 0.25, 1.0 / 9.0]), "loss applies 1/h^gamma along decoder horizon", f"unexpected loss ratios: {ratios.tolist()}")


def audit_deliverables(audit: Audit) -> None:
    required = [
        PAPER_DIR / "main.tex",
        PAPER_DIR / "main.pdf",
        PAPER_DIR / "supplementary.tex",
        PAPER_DIR / "supplementary.pdf",
        PAPER_DIR / "cover_letter.tex",
        PAPER_DIR / "cover_letter.pdf",
        PAPER_DIR / "highlights.txt",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    audit.check(not missing, "all editable and PDF deliverables exist", f"missing deliverables: {missing}")
    if (PAPER_DIR / "main.tex").exists():
        main_tex = (PAPER_DIR / "main.tex").read_text(encoding="utf-8")
        audit.check(
            r"\begin{highlights}" not in main_tex,
            "highlights are supplied separately rather than duplicated in main.tex",
            "main.tex contains a highlights block that should be uploaded separately",
        )
    for path in (PAPER_DIR / "main.pdf", PAPER_DIR / "supplementary.pdf", PAPER_DIR / "cover_letter.pdf"):
        if path.exists():
            audit.check(path.read_bytes()[:4] == b"%PDF" and path.stat().st_size > 10_000, f"{path.name} is a non-empty PDF", f"{path.name} is invalid or unexpectedly small")
            try:
                reader = PdfReader(path)
                extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
                audit.check(bool(extracted.strip()), f"{path.name} text extracts successfully", f"{path.name} has no extractable text")
                bad_text = any(
                    re.search(pattern, extracted, flags=re.IGNORECASE)
                    for pattern in PLACEHOLDERS
                ) or "\ufffd" in extracted or "â" in extracted
                audit.check(not bad_text, f"{path.name} extracted text has no placeholders or encoding damage", f"{path.name} extracted text contains a placeholder or damaged character")
                metadata = reader.metadata or {}
                audit.check(bool(metadata.get("/Title")) and bool(metadata.get("/Author")), f"{path.name} has title and author metadata", f"{path.name} lacks title or author metadata")
                if path.name == "cover_letter.pdf":
                    audit.check(len(reader.pages) == 1, "cover letter is one page", f"cover letter is {len(reader.pages)} pages")
                if path.name == "main.pdf":
                    audit.check(
                        len(reader.pages) <= 20,
                        f"main.pdf meets the 20-page preference ({len(reader.pages)} pages)",
                        f"main.pdf is {len(reader.pages)} pages; the KBS guide prefers at most 20 including figures",
                    )
            except Exception as exc:
                audit.failures.append(f"could not parse {path.name}: {exc}")


def main() -> int:
    audit = Audit()
    text = MANUSCRIPT.read_text(encoding="utf-8")
    audit_manuscript(audit, text)
    audit_placeholders(audit)
    audit_results(audit)
    audit_reported_tables(audit, text)
    audit_secondary_claims(audit)
    audit_loss(audit)
    audit_deliverables(audit)
    for item in audit.passes:
        print(f"PASS: {item}")
    for item in audit.warnings:
        print(f"WARNING: {item}")
    if audit.failures:
        for item in audit.failures:
            print(f"FAIL: {item}", file=sys.stderr)
        print(f"\nFAILED: {len(audit.failures)} publication check(s)", file=sys.stderr)
        return 1
    print(f"\nPASSED: {len(audit.passes)} publication checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
