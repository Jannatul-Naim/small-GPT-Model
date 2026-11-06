"""Turn a Claude Doc's markdown export into a LaTeX file in Documentation/.

Usage: python reports/doc_to_latex.py learning|project <export.md> <tool-results dir> <out dir>
Figures are widget screenshots saved in the tool-results folder, matched by their stem.
"""
import html
import re
import shutil
import sys
from pathlib import Path

DOCS = {
    "learning": {
        "tex": "How-an-LLM-Works.tex",
        "figdir": "how-an-llm-works-figures",
        "title": r"How an LLM Works,\\Through the Small-GPT Code",
        "date": "3 November 2026",
        "newpage_before": "Papers",
        "figures": [
            ("1793701705800", "pipeline"),
            ("1793701961514", "attention-he"),
            ("1793702099238", "gqa-flow"),
            ("1793702100240", "transformer-block"),
            ("1793702283221", "lr-schedule"),
            ("1793702282915", "loss-curve"),
        ],
    },
    "project": {
        "tex": "Small-GPT-Project-Document.tex",
        "figdir": "small-gpt-project-figures",
        "title": r"Small-GPT\\Project Document",
        "date": "6 November 2026",
        "newpage_before": None,
        "figures": [("1793938486890", "v3-loss-curve")],
    },
}

DOC = DOCS[sys.argv[1]]
SOURCE, RESULTS, OUT = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
FIGDIR = DOC["figdir"]
FIGURES = DOC["figures"]

PREAMBLE = r"""% Compile with XeLaTeX or Tectonic: tectonic @TEX@
\documentclass[11pt,a4paper]{article}
\usepackage{iftex}
\ifPDFTeX
  \usepackage[utf8]{inputenc}
  \usepackage[T1]{fontenc}
  \usepackage{lmodern}
\else
  \usepackage{fontspec}
\fi
\usepackage[margin=2.4cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{microtype}
\usepackage{parskip}
\usepackage{graphicx}
\usepackage{float}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage{array,booktabs,longtable}
\usepackage{xcolor}
\usepackage{listings}
\usepackage{needspace}
\usepackage[colorlinks=true,linkcolor=black,urlcolor=blue!55!black]{hyperref}

\definecolor{codebg}{gray}{0.965}
\definecolor{coderule}{gray}{0.82}
\lstset{
  basicstyle=\ttfamily\footnotesize,
  backgroundcolor=\color{codebg},
  frame=single,
  rulecolor=\color{coderule},
  framesep=5pt,
  xleftmargin=6pt,
  xrightmargin=6pt,
  columns=fullflexible,
  keepspaces=true,
  upquote=true,
  breaklines=true,
  breakindent=18pt,
  showstringspaces=false,
  keywordstyle=\color{blue!60!black},
  commentstyle=\color{black!55},
  stringstyle=\color{green!35!black},
  aboveskip=8pt,
  belowskip=8pt,
}
\setcounter{tocdepth}{1}
\setlength{\LTpre}{6pt}
\setlength{\LTpost}{8pt}
\newcommand{\vs}{\textvisiblespace}
\newcommand{\code}[1]{\texttt{#1}}
\raggedbottom

\title{@TITLE@}
\author{}
\date{@DATE@}

\begin{document}
\maketitle
"""
PREAMBLE = PREAMBLE.replace("@TEX@", DOC["tex"]).replace("@TITLE@", DOC["title"]).replace("@DATE@", DOC["date"])
TOC = "\\tableofcontents\n\\newpage"

SYMBOLS = {
    "\u2423": r"\vs{}",
    "\u00d7": r"$\times$",
    "\u2212": r"$-$",
    "\u00b7": r"$\cdot$",
    "\u00f7": r"$\div$",
    "\u00b2": r"\textsuperscript{2}",
    "\u03b1": r"$\alpha$",
    "\u03b2": r"$\beta$",
    "\u03b7": r"$\eta$",
    "\u03b8": r"$\theta$",
    "\u03bb": r"$\lambda$",
}

# Math written as plain text in the guide. Longest forms first.
MATH = [
    ("4096^768", r"$4096^{768}$"),
    ("10000^(2i/d)", r"$10000^{2i/d}$"),
    ("\u221av\u0302", r"$\sqrt{\hat{v}}$"),
    ("m\u0302", r"$\hat{m}$"),
    ("v\u0302", r"$\hat{v}$"),
    ("\u03b8\u2032", r"$\theta'$"),
    ("\u03b2\u2081", r"$\beta_1$"),
    ("\u03b2\u2082", r"$\beta_2$"),
    ("W_Q", "$W_Q$"),
    ("W_K", "$W_K$"),
    ("W_V", "$W_V$"),
    ("W_O", "$W_O$"),
    ("d_k", "$d_k$"),
    ("S_w", "$S_w$"),
    ("y_t", "$y_t$"),
]

ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def unescape_md(text):
    return html.unescape(re.sub(r"\\([\\`*_{}\[\]()#+\-.!|<>~^=])", r"\1", text))


def plain(text):
    """Escape ordinary text, keeping the math and symbols readable."""
    text = unescape_md(text)
    text = re.sub(r'(^|[\s(\[])"', "\\1\u201c", text).replace('"', "\u201d")
    parts = re.split("(" + "|".join(re.escape(m[0]) for m in MATH) + ")", text)
    lookup = dict(MATH)
    out = []
    for part in parts:
        if part in lookup:
            out.append(lookup[part])
            continue
        for ch in part:
            out.append(ESCAPES.get(ch) or SYMBOLS.get(ch) or ch)
    return "".join(out)


def code_span(text):
    out = []
    for ch in text:
        if ch in ESCAPES:
            out.append(ESCAPES[ch])
        else:
            out.append(ch)
        if ch in "_./,-":
            out.append(r"\allowbreak{}")
    return r"\code{" + "".join(out) + "}"


INLINE = re.compile(r"(`[^`]+`|\*\*(?:[^*]|\*(?!\*))+\*\*|(?<!\\)\[(?:[^\]\\]|\\.)+\]\([^)\s]+\))")


def inline(text):
    out = []
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("`"):
            out.append(code_span(part[1:-1]))
        elif part.startswith("**"):
            out.append(r"\textbf{" + inline(part[2:-2]) + "}")
        elif INLINE.fullmatch(part):
            label, url = re.fullmatch(r"\[(.+)\]\((.+)\)", part, re.S).groups()
            url = url.replace("%", r"\%").replace("#", r"\#")
            out.append(r"\href{" + url + "}{" + inline(label) + "}")
        else:
            out.append(plain(part))
    return "".join(out)


def heading_text(text):
    tex = inline(text)
    return re.sub(r"\$([^$]+)\$", lambda m: r"\texorpdfstring{$" + m[1] + "$}{" + m[1].replace("_", "") + "}", tex)


def cells(line):
    line = line.strip()
    line = line[1:] if line.startswith("|") else line
    line = line[:-1] if line.endswith("|") and not line.endswith(r"\|") else line
    return [c.strip() for c in re.split(r"(?<!\\)\|", line)]


def table(rows):
    header, body = rows[0], rows[2:]
    n = len(header)
    # Points per character at each size, roughly, and the text width in points.
    size, char_pt = (r"\footnotesize", 5.0) if n >= 7 else (r"\small", 5.2)
    usable = 459 - 12 * n
    need, want = [], []
    for i in range(n):
        column = [r[i] if i < len(r) else "" for r in [header] + body]
        # Each column must fit its longest unbreakable word; code spans and hyphens can break.
        words = []
        for j, c in enumerate(column):
            c = re.sub(r"\]\([^)]*\)", "]", c)
            c = re.sub(r"`[^`]*`", lambda m: re.sub(r"[_./,-]", " ", m[0]), c)
            scale = 1.18 if j == 0 else 1.0  # the header is bold
            words += [len(w) * scale for w in re.split(r"[\s-]+", c.replace("**", "").replace("`", "")) if w]
        need.append(max(words, default=1) * char_pt)
        want.append(min(max((len(c) for c in column[1:]), default=1), 62))  # headers can wrap
    spare = max(usable - sum(need), 0)
    widths = [nd + 12 + spare * w / sum(want) for nd, w in zip(need, want)]
    total = sum(widths)
    spec = "".join(
        r">{\raggedright\arraybackslash\hspace{0pt}}p{\dimexpr %.3f\linewidth-2\tabcolsep\relax}" % (w / total)
        for w in widths
    )
    lines = ["{" + size, r"\begin{longtable}{@{}" + spec.replace(r"\dimexpr", r"\dimexpr") + "@{}}", r"\toprule"]
    lines.append(" & ".join(r"\textbf{" + inline(c) + "}" if c else "" for c in header) + r" \\")
    lines += [r"\midrule", r"\endhead"]
    for row in body:
        row = row + [""] * (n - len(row))
        lines.append(" & ".join(inline(c) for c in row[:n]) + r" \\")
    lines += [r"\bottomrule", r"\end{longtable}", "}"]
    return "\n".join(lines)


def section(level, text):
    command = {2: "section", 3: "subsection"}[level]
    tex = heading_text(text)
    return f"\\{command}*{{{tex}}}\n\\phantomsection\n\\addcontentsline{{toc}}{{{command}}}{{{tex}}}"


def convert(md):
    lines = md.split("\n")
    out = [PREAMBLE]
    i = 0
    figure = 0
    toc_done = False

    while i < len(lines):
        line = lines[i]

        if line.startswith("```"):
            language = line[3:].strip()
            i += 1
            block = []
            while not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            body = "\n".join(block)
            if language == "latex":
                # Same paragraph as the sentence before it, so a page break
                # cannot separate a formula from its introduction.
                if out[-1].startswith("\\[") or out[-1][-1].isalnum() or out[-1].endswith((":", ".", "}")):
                    out[-1] += "\n\\[\n" + body + "\n\\]"
                else:
                    out.append("\\[\n" + body + "\n\\]")
            else:
                name = {"python": "Python"}.get(language)  # bash stays plain, so --eval-interval is not coloured
                option = f"[language={name}]" if name else ""
                room = min(int(len(block) * 0.85) + 3, 34)
                listing = f"\\begin{{lstlisting}}{option}\n{body}\n\\end{{lstlisting}}"
                if out[-1].startswith(("\\section*", "\\subsection*")):
                    # Reserve room before the heading, so it stays with its code.
                    out[-1] = f"\\Needspace{{{room + 4}\\baselineskip}}\n" + out[-1]
                    out.append(listing)
                else:
                    out.append(f"\\Needspace{{{room}\\baselineskip}}\n" + listing)
            continue

        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(cells(lines[i]))
                i += 1
            if out[-1].startswith(("\\section*", "\\subsection*")):
                out[-1] = "\\Needspace{10\\baselineskip}\n" + out[-1]
            out.append(table(rows))
            continue

        match = re.match(r"(#{1,3}) (.*)", line)
        if match:
            level = len(match[1])
            if level > 1:
                if not toc_done:  # no lead paragraph, so the contents go before the first section
                    toc_done = True
                    out.append(TOC)
                if DOC["newpage_before"] and match[2].startswith(DOC["newpage_before"]):
                    out.append("\\newpage")
                out.append(section(level, match[2]))
            i += 1
            continue

        if re.match(r"(- |\d+\. )", line):
            ordered = bool(re.match(r"\d+\. ", line))
            start = int(line.split(".")[0]) if ordered else 1
            items = []
            while i < len(lines) and (re.match(r"(- |\d+\. )", lines[i]) or not lines[i].strip()):
                if lines[i].strip():
                    items.append(re.sub(r"^(- |\d+\. )", "", lines[i]))
                else:
                    following = lines[i + 1] if i + 1 < len(lines) else ""
                    if not re.match(r"(- |\d+\. )", following):
                        break
                i += 1
            env = "enumerate" if ordered else "itemize"
            # A numbered list split by a code block carries on from its number.
            counter = f"\\setcounter{{enumi}}{{{start - 1}}}\n" if start > 1 else ""
            body = []
            for t in items:
                if t.startswith("[ ] "):  # a task
                    body.append(r"\item[$\square$] " + inline(t[4:]))
                else:
                    body.append(r"\item " + inline(t))
            out.append(f"\\begin{{{env}}}\n" + counter + "\n".join(body) + f"\n\\end{{{env}}}")
            continue

        match = re.match(r"(?:&#91;|\\\[|\[)embedded content: (.*?)\\?\]\s*$", line)
        if match:
            stem, name = FIGURES[figure]
            figure += 1
            out.append(
                "\\begin{figure}[H]\n\\centering\n"
                f"\\includegraphics[width=\\linewidth]{{{FIGDIR}/{name}.png}}\n"
                f"\\caption{{{inline(match[1])}}}\n\\end{{figure}}"
            )
            i += 1
            continue

        if line.strip():
            if re.match(r"[A-Z][a-z]{2} \d{1,2}, \d{4}", line):  # the byline; the title page has the date
                i += 1
                continue
            paragraph = inline(line)
            if not toc_done:  # a lead paragraph sits above the contents
                toc_done = True
                out.append(paragraph + "\n\n" + TOC)
            else:
                out.append(paragraph)
        i += 1

    assert figure == len(FIGURES), figure
    out.append("\\end{document}\n")
    return "\n\n".join(out)


tex = convert(SOURCE.read_text(encoding="utf-8"))
(OUT / FIGDIR).mkdir(parents=True, exist_ok=True)
for stem, name in FIGURES:
    found = list(RESULTS.glob(f"*blob-{stem}-*.png"))
    shutil.copyfile(found[0], OUT / FIGDIR / f"{name}.png")
target = OUT / DOC["tex"]
tmp = target.with_suffix(".tex.tmp")
tmp.write_text(tex, encoding="utf-8")
tmp.replace(target)

left = sorted({ch for ch in tex if ord(ch) > 127})
print("non-ascii left:", left)
print("lines", tex.count("\n"), "sections", tex.count("\\section*"), "subsections", tex.count("\\subsection*"))
print("math", tex.count("\\["), "listings", tex.count("\\begin{lstlisting}"), "tables", tex.count("\\begin{longtable}"), "figures", tex.count("\\begin{figure}"))
