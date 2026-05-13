#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript" / "manuscript.md"
OUTPUT = ROOT / "manuscript" / "springer_submission.tex"


def protect_inline(text: str):
    replacements = {}

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
    replacements = {
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
    return "".join(replacements.get(char, char) for char in text)


def convert_inline(text: str) -> str:
    protected, replacements = protect_inline(text)
    return restore_inline(escape_latex(protected), replacements)


def convert_blocks(markdown: str) -> str:
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
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            close_list()
            output.append(r"\section{" + convert_inline(clean_heading(line[3:])) + "}")
            continue
        if line.startswith("### "):
            close_list()
            output.append(r"\subsection{" + convert_inline(clean_heading(line[4:])) + "}")
            continue
        if line.startswith("- "):
            if list_mode != "itemize":
                close_list()
                output.append(r"\begin{itemize}")
                list_mode = "itemize"
            output.append(r"\item " + convert_inline(line[2:]))
            continue
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
    return "\n".join(output).strip() + "\n"


def clean_heading(text: str) -> str:
    return re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", text)


def main() -> int:
    markdown = MANUSCRIPT.read_text(encoding="utf-8")
    title_match = re.search(r"^# (.+)$", markdown, re.MULTILINE)
    title = title_match.group(1) if title_match else "Manuscript"
    body = re.sub(r"^# .+\n+", "", markdown, count=1)
    abstract_match = re.search(r"## Abstract\n\n(.*?)\n\nKeywords:\s*(.*?)\n\n", body, re.S)
    abstract = abstract_match.group(1).strip() if abstract_match else ""
    keywords = abstract_match.group(2).strip() if abstract_match else ""
    if abstract_match:
        body = body[: abstract_match.start()] + body[abstract_match.end() :]

    latex = rf"""\documentclass[sn-mathphys-num]{{sn-jnl}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{natbib}}

\jyear{{2026}}

\title[{convert_inline(title)}]{{{convert_inline(title)}}}

\author*[1]{{\fnm{{[Author 1]}} \sur{{[Surname]}}}}\email{{[email]}}
\affil*[1]{{\orgdiv{{[Department]}}, \orgname{{[Institution]}}, \country{{[Country]}}}}

\abstract{{{convert_inline(abstract)}}}

\keywords{{{convert_inline(keywords)}}}

\begin{{document}}

\maketitle

{convert_blocks(body)}

\bibliography{{references}}

\end{{document}}
"""
    OUTPUT.write_text(latex, encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
