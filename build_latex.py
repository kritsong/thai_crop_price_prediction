# -*- coding: utf-8 -*-
"""Convert final_paper.md to an elsarticle LaTeX project (main.tex)."""
import re, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from paths import PAPER_DIR

SRC = str(PAPER_DIR / "final_paper.md")
OUT = str(PAPER_DIR / "main.tex")
lines = open(SRC, encoding="utf-8").read().split("\n")

UNI = {
    "—": "---", "–": "--", "−": "-",
    "≥": r"\ensuremath{\geq}", "≤": r"\ensuremath{\leq}",
    "≈": r"\ensuremath{\approx}", "×": r"\ensuremath{\times}",
    "±": r"\ensuremath{\pm}", "→": r"\ensuremath{\rightarrow}",
    "“": "``", "”": "''", "‘": "`", "’": "'",
    "…": r"\ldots{}", " ": "~", "ł": r"\l{}",
    "ç": r"\c{c}", "ü": r'\"u', "ö": r'\"o', "ı": r"\i{}",
    "å": r"\aa{}", "ş": r"\c{s}", "é": r"\'e", "ú": r"\'u",
}

def esc_specials(s):
    # escape LaTeX specials + map unicode; operates on a NON-math text fragment
    s = s.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#").replace("_", r"\_")
    for u, r in UNI.items():
        s = s.replace(u, r)
    return s

def inline(s):
    codes = []
    urls = []
    # Protect bare URLs from escaping and render them with break opportunities.
    def repl_url(m):
        value = m.group(0)
        suffix = ""
        while value and value[-1] in ".,":
            suffix = value[-1] + suffix
            value = value[:-1]
        urls.append(value)
        return "\x01%d\x01%s" % (len(urls) - 1, suffix)
    s = re.sub(r"https?://[^\s]+", repl_url, s)
    # 1. protect `code` spans (they never cross math)
    def repl_code(m):
        codes.append(m.group(1))
        return "\x00%d\x00" % (len(codes) - 1)
    s = re.sub(r"`([^`]*)`", repl_code, s)
    # 2. bold / italic on the FULL line, so a span may wrap inline $math$
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", s)
    # 3. split on $, escape only the non-math (even) parts
    parts = s.split("$")
    out = []
    for k, p in enumerate(parts):
        out.append(esc_specials(p) if k % 2 == 0 else "$" + p + "$")
    s = "".join(out)
    # 4. restore code as \texttt, escaping its content
    def back(m):
        c = codes[int(m.group(1))]
        c = c.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")
        return r"\texttt{%s}" % c
    s = re.sub("\x00(\d+)\x00", back, s)
    return re.sub("\x01(\d+)\x01", lambda m: r"\url{%s}" % urls[int(m.group(1))], s)

# ---- locate blocks ----
def idx(pred):
    for i, l in enumerate(lines):
        if pred(l):
            return i
    return -1

i_title = 0
title = lines[0].lstrip("# ").strip()
i_hl = idx(lambda l: l.strip() == "## Highlights")
i_abs = idx(lambda l: l.strip() == "## Abstract")
i_kw = idx(lambda l: l.startswith("**Keywords:**"))
i_intro = idx(lambda l: l.strip() == "## 1. Introduction")
i_cred = idx(lambda l: l.strip() == "## CRediT authorship contribution statement")
i_refs = idx(lambda l: l.strip() == "## References")

highlights = [l[2:].strip() for l in lines[i_hl + 1:i_abs] if l.strip().startswith("- ")]
abstract = [l.strip() for l in lines[i_abs + 1:i_kw] if l.strip() and not l.startswith("#")]
keywords = lines[i_kw].replace("**Keywords:**", "").strip()
kw_list = [k.strip() for k in keywords.split(";") if k.strip()]

# ---- body converter (sections 1..8.1) ----
def convert_body(body):
    out = []
    n = len(body)
    i = 0
    list_stack = []  # 'itemize'/'enumerate'

    def close_lists():
        while list_stack:
            out.append("\\end{%s}" % list_stack.pop())

    while i < n:
        raw = body[i]
        s = raw.rstrip()
        st = s.strip()

        # blank — do NOT close lists (multi-paragraph items rely on this);
        # lists are closed by headings/figures/tables or non-continuation paragraphs
        if st == "":
            out.append("")
            i += 1
            continue
        # horizontal rule
        if st == "---":
            i += 1
            continue
        # figure
        m = re.match(r"!\[.*?\]\(images/(.+?)\)", st)
        if m:
            close_lists()
            fn = m.group(1)
            cap = ""
            j = i + 1
            while j < n and body[j].strip() == "":
                j += 1
            if j < n:
                cm = re.match(r"\*(.+)\*$", body[j].strip())
                if cm:
                    # strip the "Figure N:" prefix; \caption adds its own label
                    cap = inline(re.sub(r"^Figure\s+\d+:\s*", "", cm.group(1)))
                    i = j
            out.append(r"\begin{figure}[htbp]\centering")
            out.append(r"\includegraphics[width=\linewidth]{%s}" % fn)
            if cap:
                out.append(r"\caption{%s}" % cap)
            out.append(r"\end{figure}")
            i += 1
            continue
        # table
        mt = re.match(r"####\s+Table\s+\d+:\s*(.+)$", st)
        if mt:
            close_lists()
            cap = inline(mt.group(1))
            # gather pipe rows
            rows = []
            j = i + 1
            while j < n and body[j].strip().startswith("|"):
                rows.append(body[j].strip())
                j += 1
            # parse
            def cells(r):
                r = r.strip()
                if r.startswith("|"): r = r[1:]
                if r.endswith("|"): r = r[:-1]
                return [c.strip() for c in r.split("|")]
            header = cells(rows[0])
            align_row = cells(rows[1])
            aligns = []
            for a in align_row:
                if a.startswith(":") and a.endswith(":"): aligns.append("c")
                elif a.endswith(":"): aligns.append("r")
                else: aligns.append("l")
            body_rows = [cells(r) for r in rows[2:]]
            out.append(r"\begin{table}[htbp]\centering")
            out.append(r"\caption{%s}" % cap)
            out.append(r"\small")
            out.append(r"\begin{tabular}{%s}" % "".join(aligns))
            out.append(r"\toprule")
            out.append(" & ".join(inline(c) for c in header) + r" \\")
            out.append(r"\midrule")
            for br in body_rows:
                out.append(" & ".join(inline(c) for c in br) + r" \\")
            out.append(r"\bottomrule")
            out.append(r"\end{tabular}")
            out.append(r"\end{table}")
            i = j
            continue
        # headings
        mh = re.match(r"(#{2,4})\s+(.+)$", st)
        if mh:
            close_lists()
            hashes, txt = mh.group(1), mh.group(2).strip()
            # strip leading "N. " or "N.N. " numbering
            txt_clean = re.sub(r"^\d+(\.\d+)*\.?\s+", "", txt)
            txt_clean = inline(txt_clean)
            if hashes == "##":
                out.append(r"\section{%s}" % txt_clean)
            elif hashes == "###":
                out.append(r"\subsection{%s}" % txt_clean)
            else:
                out.append(r"\subsubsection{%s}" % txt_clean)
            i += 1
            continue
        # enumerated list item:  "N. text"
        me = re.match(r"(\d+)\.\s+(.+)$", s)
        if me:
            if not list_stack or list_stack[-1] != "enumerate":
                close_lists()
                out.append(r"\begin{enumerate}")
                list_stack.append("enumerate")
            out.append(r"\item " + inline(me.group(2)))
            i += 1
            continue
        # bullet item: "*   text" or "- text"
        mb = re.match(r"[*\-]\s+(.+)$", s)
        if mb and not st.startswith("**"):
            if not list_stack or list_stack[-1] != "itemize":
                close_lists()
                out.append(r"\begin{itemize}")
                list_stack.append("itemize")
            out.append(r"\item " + inline(mb.group(1)))
            i += 1
            continue

        # --- non-list content (display math or plain paragraph) ---
        def is_continuation(start, ltype):
            # is there another list item of ltype before the next heading/figure/table?
            j = start
            while j < n:
                t = body[j].strip()
                if t == "":
                    j += 1; continue
                if re.match(r"#{2,4}\s", t) or re.match(r"!\[.*?\]\(images/", t):
                    return False
                r = body[j].rstrip()
                if ltype == "enumerate" and re.match(r"\d+\.\s", r):
                    return True
                if ltype == "itemize" and re.match(r"[*\-]\s", r) and not t.startswith("**"):
                    return True
                j += 1
            return False

        # an indented line always continues the currently open list item, even
        # when it belongs to the final item (where the lookahead finds no
        # further "N. " marker and would otherwise close the list too early)
        indented = bool(re.match(r"\s{2,}\S", raw))
        if list_stack and not indented and not is_continuation(i, list_stack[-1]):
            close_lists()

        if st.startswith("$$") and st.endswith("$$") and len(st) > 4:
            out.append(r"\[" + st[2:-2].strip() + r"\]")
        else:
            out.append(inline(s))
        i += 1

    close_lists()
    return "\n".join(out)

body_lines = lines[i_intro:i_cred]
body_tex = convert_body(body_lines)

# declarations (CRediT .. Data availability) -> unnumbered sections
decl_lines = lines[i_cred:i_refs]
def convert_decls(dl):
    out = []
    for l in dl:
        st = l.strip()
        if st.startswith("## "):
            out.append(r"\section*{%s}" % inline(st[3:]))
        elif st == "" or st == "---":
            out.append("")
        else:
            out.append(inline(st))
    return "\n".join(out)
decl_tex = convert_decls(decl_lines)

# references
ref_lines = [l.strip() for l in lines[i_refs + 1:] if l.strip()]
ref_items = [r for r in ref_lines if not r.startswith("*Reference details")]

# ---- assemble ----
def esc_plain(s):
    return inline(s)

tex = []
tex.append(r"\documentclass[review,3p,times]{elsarticle}")
tex.append(r"\usepackage[utf8]{inputenc}")
tex.append(r"\usepackage{amsmath,amssymb}")
tex.append(r"\usepackage{booktabs}")
tex.append(r"\usepackage{graphicx}")
tex.append(r"\usepackage[hidelinks,hypertexnames=false,bookmarks=false]{hyperref}")
tex.append(r"\graphicspath{{images/}}")
tex.append(r"\usepackage{array}")
tex.append(r"\setlength{\emergencystretch}{2em}")
tex.append(r"\journal{Knowledge-Based Systems}")
tex.append(r"\hypersetup{pdftitle={Beyond the Random Walk: Horizon-Weighted Temporal Fusion Transformers for Thai Agricultural Commodity Price Forecasting},pdfauthor={Kritaphat Songsri-in, Auyporn Chukeaw, Munlika Rattaphun, Walaiporn Sornkliang, Rattayagon Thaiphan}}")
tex.append(r"\begin{document}")
tex.append(r"\begin{frontmatter}")
tex.append(r"\title{%s}" % esc_plain(title))
# Author names and affiliation follow the manuscript. The corresponding-author
# address and email are consistent with the author's recent publications.
tex.append(r"\author[a]{Kritaphat Songsri-in\corref{cor1}}")
tex.append(r"\author[a]{Auyporn Chukeaw}")
tex.append(r"\author[a]{Munlika Rattaphun}")
tex.append(r"\author[a]{Walaiporn Sornkliang}")
tex.append(r"\author[a]{Rattayagon Thaiphan}")
tex.append(r"\affiliation[a]{organization={Department of Computer Science, Faculty of Science and Technology, "
           r"Nakhon Si Thammarat Rajabhat University}, addressline={1 Moo 4, Tha Ngio}, "
           r"city={Mueang Nakhon Si Thammarat}, postcode={80280}, state={Nakhon Si Thammarat}, country={Thailand}}")
tex.append(r"\cortext[cor1]{Corresponding author. E-mail: kritaphat\_son@nstru.ac.th}")
tex.append(r"\begin{abstract}")
for p in abstract:
    tex.append(esc_plain(p))
    tex.append("")
tex.append(r"\end{abstract}")
tex.append(r"\begin{highlights}")
for h in highlights:
    tex.append(r"\item " + esc_plain(h))
tex.append(r"\end{highlights}")
tex.append(r"\begin{keyword}")
tex.append(" \\sep ".join(esc_plain(k) for k in kw_list))
tex.append(r"\end{keyword}")
tex.append(r"\end{frontmatter}")
tex.append("")
tex.append(body_tex)
tex.append("")
tex.append(decl_tex)
tex.append("")
tex.append(r"\section*{References}")
tex.append(r"\begingroup")
tex.append(r"\setlength{\parindent}{0pt}")
for r in ref_items:
    tex.append(r"\hangindent=1.5em\hangafter=1 " + esc_plain(r) + r"\par\medskip")
tex.append(r"\endgroup")
tex.append(r"\end{document}")

open(OUT, "w", encoding="utf-8").write("\n".join(tex))
print("wrote", OUT)
print("highlights:", len(highlights), "| abstract paras:", len(abstract), "| kw:", len(kw_list), "| refs:", len(ref_items))
