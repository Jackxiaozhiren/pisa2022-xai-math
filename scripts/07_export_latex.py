#!/usr/bin/env python3
"""Convert manuscript.md + tables to Springer Nature LaTeX submission."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript" / "manuscript.md"
OUTPUT = ROOT / "manuscript" / "springer_submission.tex"


def protect_inline(text: str) -> tuple[str, dict[str, str]]:
    replacements: dict[str, str] = {}

    def store(prefix: str, value: str) -> str:
        key = f"@@{prefix}{len(replacements)}@@"
        replacements[key] = value
        return key

    def cite_repl(match):
        keys = re.findall(r"@([A-Za-z0-9_:-]+)", match.group(1))
        return store("CITE", r"\citep{" + ",".join(keys) + "}")

    def code_repl(match):
        return store("CODE", r"\texttt{" + escape_latex(match.group(1)) + "}")

    text = re.sub(r"\[([^\]]*@+[^\]]*)\]", cite_repl, text)
    text = re.sub(r"`([^`]+)`", code_repl, text)
    return text, replacements


def restore_inline(text: str, replacements: dict[str, str]) -> str:
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text


def escape_latex(text: str) -> str:
    chars = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(chars.get(c, c) for c in text)


def convert_inline(text: str) -> str:
    protected, replacements = protect_inline(text)
    return restore_inline(escape_latex(protected), replacements)


def clean_heading(text: str) -> str:
    return re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", text)


def convert_body(markdown: str) -> str:
    lines = markdown.splitlines()
    output: list[str] = []
    list_mode: str | None = None

    def close_list():
        nonlocal list_mode
        if list_mode == "itemize":
            output.append(r"\end{itemize}")
        elif list_mode == "enumerate":
            output.append(r"\end{enumerate}")
        list_mode = None

    for raw in lines:
        line = raw.rstrip()
        if not line:
            close_list()
            output.append("")
            continue

        # Skip heading markers that are handled by header/preamble
        if line.startswith("# "):
            continue

        if line.startswith("## "):
            close_list()
            heading = clean_heading(line[3:])
            output.append(r"\section{" + convert_inline(heading) + "}")
            continue

        if line.startswith("### "):
            close_list()
            heading = clean_heading(line[4:])
            output.append(r"\subsection{" + convert_inline(heading) + "}")
            continue

        # Handle figure images
        img_match = re.match(r"!\[(.*?)\]\((.*?)\)", line)
        if img_match:
            close_list()
            caption = img_match.group(1)
            path = img_match.group(2)
            continue  # figures are manually placed

        # Handle bullet lists
        if line.startswith("- "):
            if list_mode != "itemize":
                close_list()
                output.append(r"\begin{itemize}")
                list_mode = "itemize"
            output.append(r"\item " + convert_inline(line[2:]))
            continue

        # Handle numbered lists
        if re.match(r"^\d+\.\s+", line):
            if list_mode != "enumerate":
                close_list()
                output.append(r"\begin{enumerate}")
                list_mode = "enumerate"
            output.append(r"\item " + convert_inline(re.sub(r"^\d+\.\s+", "", line)))
            continue

        close_list()
        output.append(convert_inline(line))

    close_list()
    return "\n".join(output).strip()


def main() -> int:
    markdown = MANUSCRIPT.read_text(encoding="utf-8")

    # Extract title
    title_match = re.search(r"^# (.+)$", markdown, re.MULTILINE)
    title = title_match.group(1) if title_match else "Manuscript"
    short_title = "XAI for PISA 2022 Mathematics Literacy"

    # Remove title line
    body = re.sub(r"^# .+\n+", "", markdown, count=1)

    # Extract abstract and keywords
    abstract = ""
    keywords = ""
    abs_match = re.search(r"## Abstract\n\n(.*?)\n\n(?:Keywords|##)", body, re.S)
    if abs_match:
        abstract = abs_match.group(1).strip()
        body = body[: abs_match.start()] + body[abs_match.end() :]

    kw_match = re.search(r"Keywords:\s*(.*?)\n\n##", body, re.S)
    if not kw_match:
        kw_match = re.search(r"Keywords:\s*(.*?)\n\n", body, re.S)
    if kw_match:
        keywords = kw_match.group(1).strip()
        body = body[: kw_match.start()] + body[kw_match.end() :]

    # Remove the "Keywords:" line if it wasn't matched with ## after
    body = re.sub(r"^Keywords:.*\n\n", "", body, flags=re.MULTILINE)

    # Remove "## Generated Tables and Figures" section and everything after
    body = re.sub(r"\n## Generated Tables and Figures.*", "", body, flags=re.DOTALL)

    latex_body = convert_body(body)

    # Build the full document
    latex = rf"""\documentclass[pdflatex,sn-mathphys-ay]{{sn-jnl}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{amsmath,amssymb,amsfonts}}
\usepackage{{url}}

\title[{short_title}]{{{title}}}

\author*[1]{{\fnm{{Zhiren}} \sur{{Xiao}}}}\email{{241734106@m.gduf.edu.cn}}
\affil*[1]{{\orgname{{Guangdong University of Finance}}, \country{{China}}}}

\abstract{{{convert_inline(abstract)}}}

\keywords{{{convert_inline(keywords)}}}

\begin{{document}}

\maketitle

{latex_body}

\section*{{Tables}}

\input{{tables/table_01_sample_summary.tex}}

\clearpage

\input{{tables/table_02_descriptive.tex}}

\input{{tables/table_03_regression.tex}}

\input{{tables/table_04_classification.tex}}

\input{{tables/table_05_variable_audit.tex}}

\input{{tables/table_06_threshold.tex}}

\input{{tables/table_07_calibration.tex}}

\input{{tables/table_08_subgroup.tex}}

\input{{tables/table_09_country.tex}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.92\textwidth]{{../reports/figures/classification_lightgbm_shap_summary.png}}
\caption{{SHAP summary for the LightGBM low-performer classification model. The plot summarizes fitted-model feature contributions in the deterministic explanation sample and should be interpreted as predictive model explanation, not causal evidence.}}
\label{{fig:classification-shap}}
\end{{figure}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.92\textwidth]{{../reports/figures/regression_lightgbm_shap_summary.png}}
\caption{{SHAP summary for the LightGBM mathematics-score regression model. The plot summarizes fitted-model feature contributions in the deterministic explanation sample and should be interpreted as predictive model explanation, not causal evidence.}}
\label{{fig:regression-shap}}
\end{{figure}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.84\textwidth]{{../reports/figures/digital_feature_importance.png}}
\caption{{Digital-feature permutation importance for the best LightGBM models. ICT resources and ICT self-efficacy were the strongest digital-learning predictors in both tasks.}}
\label{{fig:digital-importance}}
\end{{figure}}

\backmatter

\section*{{Declarations}}

\bmhead{{Funding}}
No external funding was received.

\bmhead{{Competing interests}}
The author declares no competing interests.

\bmhead{{Ethics approval and consent to participate}}
This study uses publicly available, de-identified secondary data from the OECD PISA 2022 Database. No new human participants were recruited by the author, and no individual-level identifiable data were accessed. Institution-specific public-data or secondary-data exemption wording remains to be confirmed before submission.

\bmhead{{Consent for publication}}
Not applicable.

\bmhead{{Data availability}}
The data are publicly available from the OECD PISA 2022 Database. Analysis code and non-restricted derived outputs will be made available in a public repository subject to OECD data-use terms and journal policy. The repository should not redistribute OECD raw data files.

\bmhead{{Code availability}}
Code, configuration files, manuscript source, and non-restricted aggregate result tables and figures will be prepared for public release at \url{{https://github.com/Jackxiaozhiren/pisa2022-xai-math}}. OECD raw data files, row-level predictions, and fitted model artifacts should not be redistributed in that public repository.

\bmhead{{Author contribution}}
XIAO ZHIREN: conceptualization, data curation, formal analysis, investigation, methodology, software, validation, visualization, writing -- original draft, writing -- review and editing.

\bmhead{{AI-assisted work}}
AI-assisted coding and drafting tools were used to support programming, reproducible analysis organization, and manuscript preparation. All intellectual decisions, interpretation, verification, and final manuscript approval remain the responsibility of the human author.

\bibliography{{references}}

\end{{document}}
"""
    OUTPUT.write_text(latex, encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
