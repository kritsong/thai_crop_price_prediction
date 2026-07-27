"""Convert paper/cover_letter.md into a submittable PDF via LaTeX.

Editorial Manager does not accept Markdown, so the cover letter needs a PDF (or
DOCX) rendering. Keeps the same house rules as the manuscript: no em dashes.
"""
import re, io, sys, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from paths import PAPER_DIR

SRC = str(PAPER_DIR / "cover_letter.md")
OUT = str(PAPER_DIR / "cover_letter.tex")

UNI = {
    "\u2013": "--", "\u2014": "---", "\u2018": "`", "\u2019": "'",
    "\u201c": "``", "\u201d": "''", "\u2265": r"$\geq$", "\u2264": r"$\leq$",
    "\u2212": "$-$", "\u00d7": r"$\times$", "\u03b3": r"$\gamma$",
}


def esc(s):
    """Escape LaTeX specials outside math spans, then restore markdown emphasis."""
    parts = s.split("$")
    out = []
    for i, p in enumerate(parts):
        if i % 2:                      # inside math
            out.append("$" + p + "$")
            continue
        for k, v in UNI.items():
            p = p.replace(k, v)
        for ch in ["\\", "&", "%", "#", "_", "{", "}"]:
            p = p.replace(ch, "\\" + ch)
        p = p.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")
        out.append(p)
    s = "".join(out)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"\\emph{\1}", s)
    s = re.sub(r"\[(.+?)\]", r"[\1]", s)
    return s


def main():
    md = open(SRC, encoding="utf-8").read()
    lines = md.split("\n")
    # drop the markdown H1, it is a file label rather than letter content
    lines = [l for l in lines if not l.strip().startswith("# ")]

    body = []
    for i, l in enumerate(lines):
        s = l.strip()
        if not s:
            body.append("")
            continue
        rendered = esc(s)
        # Body paragraphs are single long lines in the source; consecutive SHORT
        # lines are address / signature blocks and must keep their line breaks.
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        # \newline rather than \\ : a following "[...]" placeholder would be
        # parsed as \\'s optional length argument and error out.
        if nxt and len(s) < 70:
            rendered += r" \newline"
        body.append(rendered)

    # collapse into paragraphs separated by blank lines
    tex = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=2.3cm]{geometry}",
        r"\usepackage{amsmath}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{times}",
        r"\usepackage[hidelinks,bookmarks=false]{hyperref}",
        r"\hypersetup{pdftitle={Cover Letter: Horizon-Weighted Temporal Fusion Transformers for Thai Agricultural Commodity Price Forecasting},pdfauthor={Kritaphat Songsri-in}}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.55em}",
        r"\setlength{\emergencystretch}{1em}",
        r"\pagestyle{empty}",
        r"\begin{document}",
    ]
    tex.extend(body)
    tex.append(r"\end{document}")
    open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(tex))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
