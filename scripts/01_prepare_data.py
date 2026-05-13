#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pisa_xai.config import load_config, resolve_project_path
from pisa_xai.features import add_math_targets, available_columns, flatten_feature_config
from pisa_xai.io import choose_single_file, find_matching_files, load_table, write_table
from pisa_xai.pisa import math_pv_columns, replicate_weight_columns


def main() -> int:
    config = load_config()
    raw_dir = resolve_project_path(config["paths"]["raw_dir"])
    processed_dir = resolve_project_path(config["paths"]["processed_dir"])

    student_file = choose_single_file(
        find_matching_files(raw_dir, config["files"]["student_patterns"]),
        "student questionnaire",
    )
    print(f"Loading student data: {student_file}")
    student = load_table(student_file)

    school_files = find_matching_files(raw_dir, config["files"]["school_patterns"])
    if school_files:
        school_file = choose_single_file(school_files, "school questionnaire")
        print(f"Loading school data: {school_file}")
        school = load_table(school_file)
        merge_keys = [config["pisa"]["country"], config["pisa"]["school_id"]]
        if all(key in student.columns for key in merge_keys) and all(key in school.columns for key in merge_keys):
            school_suffix = "_SCH"
            data = student.merge(school, on=merge_keys, how="left", suffixes=("", school_suffix))
        else:
            print("School merge skipped because merge keys are missing.")
            data = student
    else:
        print("No school file found; preparing student-only dataset.")
        data = student

    if config["sample"]["countries"]:
        country_col = config["pisa"]["country"]
        data = data[data[country_col].isin(config["sample"]["countries"])].copy()

    max_rows = config["sample"]["max_rows_for_development"]
    if max_rows:
        data = data.sample(n=min(max_rows, len(data)), random_state=config["sample"]["random_state"])

    data = add_math_targets(
        data,
        pv_count=config["pisa"]["math_pv_count"],
        low_threshold=config["pisa"]["low_performer_threshold"],
    )

    feature_names = flatten_feature_config(config["features"])
    required = [
        config["pisa"]["country"],
        config["pisa"]["student_id"],
        config["pisa"]["school_id"],
        config["pisa"]["student_weight"],
        "MATH_PV_MEAN",
        "LOW_PERFORMER_MATH",
    ]
    required.extend(math_pv_columns(config["pisa"]["math_pv_count"]))
    required.extend(replicate_weight_columns(config["pisa"]["replicate_weight_prefix"], config["pisa"]["replicate_weight_count"]))

    available_features, missing_features = available_columns(data, feature_names)
    available_required, missing_required = available_columns(data, required)
    keep = available_required + [name for name in available_features if name not in available_required]
    prepared = data[keep].copy()

    processed_path = processed_dir / "pisa2022_math_model_frame.parquet"
    write_table(prepared, processed_path)

    report = {
        "student_file": str(student_file),
        "rows": int(len(prepared)),
        "columns": int(len(prepared.columns)),
        "available_features": available_features,
        "missing_features": missing_features,
        "missing_required": missing_required,
        "output": str(processed_path),
    }
    report_path = processed_dir / "prepare_data_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
