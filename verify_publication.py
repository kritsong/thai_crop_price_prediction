"""Fail-fast audit of the files submitted with the manuscript.

This complements ``verify_experiments.py``: it checks journal-facing limits,
submission placeholders, citation and figure integrity, the selected result
contract, and the implemented horizon weighting.  Run it after building PDFs.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pypdf import PdfReader

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
    audit.check(metrics_path.is_file(), "champion metric table exists", "missing champion metric table")
    audit.check(selection_path.is_file(), "validation selection table exists", "missing validation selection table")
    audit.check(provenance_path.is_file(), "publication provenance exists", "missing publication provenance")
    if not all(path.is_file() for path in (metrics_path, selection_path, provenance_path)):
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

    if CHAMPION_NPZ.exists():
        with np.load(CHAMPION_NPZ) as z:
            recomputed = [
                float(np.mean(np.abs(z[f"test_h{h}_actual"] - z[f"test_h{h}_median"])))
                for h in EXPECTED_HORIZONS
            ]
        audit.check(np.allclose(recomputed, metrics.MAE, atol=1e-12), "CSV metrics reproduce from champion NPZ", "CSV metrics do not reproduce from champion NPZ")
    else:
        audit.passes.append("champion NPZ not local; aggregate/provenance audit used")


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
                if path.name == "main.pdf" and len(reader.pages) > 20:
                    audit.warnings.append(
                        f"main.pdf is {len(reader.pages)} double-spaced pages; the KBS guide "
                        "prefers at most 20 including figures, but states this as a preference"
                    )
            except Exception as exc:
                audit.failures.append(f"could not parse {path.name}: {exc}")


def main() -> int:
    audit = Audit()
    text = MANUSCRIPT.read_text(encoding="utf-8")
    audit_manuscript(audit, text)
    audit_placeholders(audit)
    audit_results(audit)
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
