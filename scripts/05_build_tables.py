#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pisa_xai.config import load_config, resolve_project_path


def main() -> int:
    config = load_config()
    tables_dir = resolve_project_path(config["paths"]["tables_dir"])
    figures_dir = resolve_project_path(config["paths"]["figures_dir"])
    manuscript_dir = resolve_project_path("manuscript")
    manuscript_dir.mkdir(parents=True, exist_ok=True)
    available_tables = sorted(path.name for path in tables_dir.glob("*.csv"))
    available_figures = sorted(
        path.name for path in figures_dir.glob("*") if path.suffix.lower() in {".png", ".pdf", ".svg"}
    )
    index = ["# Generated Analysis Artifacts", "", "## Tables", ""]
    if available_tables:
        index.extend(f"- `{name}`" for name in available_tables)
    else:
        index.append("No generated tables yet.")
    index.extend(["", "## Figures", ""])
    if available_figures:
        index.extend(f"- `{name}`" for name in available_figures)
    else:
        index.append("No generated figures yet.")
    (manuscript_dir / "generated_tables_index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    print("\n".join(index))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
