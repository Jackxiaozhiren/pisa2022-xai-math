#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pisa_xai.config import load_config, resolve_project_path
from pisa_xai.io import find_matching_files
from pisa_xai.pisa import math_pv_columns, replicate_weight_columns


def main() -> int:
    config = load_config()
    raw_dir = resolve_project_path(config["paths"]["raw_dir"])
    print(f"Raw data directory: {raw_dir}")
    raw_dir.mkdir(parents=True, exist_ok=True)

    student_files = find_matching_files(raw_dir, config["files"]["student_patterns"])
    school_files = find_matching_files(raw_dir, config["files"]["school_patterns"])

    print("\nDetected student files:")
    print("\n".join(f"  - {path.name}" for path in student_files) or "  none")
    print("\nDetected school files:")
    print("\n".join(f"  - {path.name}" for path in school_files) or "  none")

    print("\nExpected core variables:")
    print(f"  country: {config['pisa']['country']}")
    print(f"  student id: {config['pisa']['student_id']}")
    print(f"  school id: {config['pisa']['school_id']}")
    print(f"  student weight: {config['pisa']['student_weight']}")
    print(f"  plausible values: {', '.join(math_pv_columns())}")
    print(f"  replicate weights: {replicate_weight_columns()[0]} ... {replicate_weight_columns()[-1]}")

    if not student_files:
        print("\nAction needed: add the PISA 2022 student questionnaire data file to data/raw/.")
        return 2
    print("\nInput check passed for the student file. School file is recommended but optional for v1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
