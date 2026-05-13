#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pisa_xai.config import load_config, resolve_project_path
from pisa_xai.features import feature_group_lookup, flatten_feature_config
from pisa_xai.io import find_matching_files, load_table
from pisa_xai.pisa import (
    brr_standard_error,
    combine_plausible_value_estimates,
    math_pv_columns,
    replicate_weight_columns,
)


FEATURE_CONSTRUCTS = {
    "ST004D01T": "gender",
    "AGE": "age",
    "GRADE": "grade placement",
    "ESCS": "economic, social, and cultural status",
    "IMMIG": "immigrant background",
    "HISEI": "parental occupational status",
    "PAREDINT": "parental education",
    "HOMEPOS": "home possessions",
    "ANXMAT": "mathematics anxiety",
    "MATHEFF": "mathematics self-efficacy",
    "MATHEF21": "mathematics self-efficacy",
    "MATHPERS": "mathematics persistence",
    "FAMCON": "family connection",
    "FAMSUP": "family support",
    "BELONG": "school belonging",
    "BULLIED": "bullying exposure",
    "FEELSAFE": "feeling safe at school",
    "SCHRISK": "school risk climate",
    "PQSCHOOL": "parent-school relationship",
    "PASCHPOL": "parental view of school policy",
    "DISCLIM": "disciplinary climate",
    "TEACHSUP": "teacher support",
    "COGACRCO": "cognitive activation",
    "COGACMCO": "cognitive activation",
    "PERFEED": "perceived feedback",
    "STUBEHA": "student behavior hindering learning",
    "TEACHBEHA": "teacher behavior hindering learning",
    "EDUSHORT": "educational material shortage",
    "STAFFSHORT": "staff shortage",
    "ICTRES": "ICT resources",
    "ICTHOME": "ICT availability or use at home",
    "ICTSCH": "ICT availability or use at school",
    "ICTEFFIC": "ICT self-efficacy",
    "ICTINFO": "ICT information behavior",
    "ICTDISTR": "digital distraction",
    "ICTSUBJ": "subject-related ICT use",
    "LEARNRES": "learning resources",
    "DISTICT": "distance-learning ICT",
    "STUDYHMW": "homework or study time",
}


def weighted_mean(df, value_col: str, weight_col: str) -> float:
    valid = df[[value_col, weight_col]].dropna()
    return float((valid[value_col] * valid[weight_col]).sum() / valid[weight_col].sum())


def weighted_rate(df, value_col: str, weight_col: str) -> float:
    return weighted_mean(df, value_col, weight_col)


def weighted_mean_series(values, weights) -> float:
    valid = values.notna() & weights.notna()
    valid_values = values.loc[valid]
    valid_weights = weights.loc[valid]
    return float((valid_values * valid_weights).sum() / valid_weights.sum())


def weighted_replicate_estimate(df, values, weight_col: str, replicate_cols: list[str]):
    full_estimate = weighted_mean_series(values, df[weight_col])
    replicate_estimates = [
        weighted_mean_series(values, df[replicate_col])
        for replicate_col in replicate_cols
        if replicate_col in df.columns
    ]
    sampling_variance = brr_standard_error(full_estimate, replicate_estimates) ** 2
    return full_estimate, sampling_variance


def ci_bounds(estimate: float, standard_error: float) -> tuple[float, float]:
    return estimate - 1.96 * standard_error, estimate + 1.96 * standard_error


def descriptive_row(
    measure: str,
    estimate: float,
    sampling_variance: float,
    imputation_variance: float,
    total_variance: float,
    note: str,
    n_unweighted: int,
    weight_col: str,
    replicate_cols: list[str],
    plausible_values: list[str],
):
    standard_error = total_variance**0.5
    ci_lower, ci_upper = ci_bounds(estimate, standard_error)
    return {
        "measure": measure,
        "estimate": estimate,
        "standard_error": standard_error,
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "sampling_variance": sampling_variance,
        "imputation_variance": imputation_variance,
        "total_variance": total_variance,
        "n_unweighted": n_unweighted,
        "weight": weight_col,
        "replicate_weights": len(replicate_cols),
        "plausible_values": ",".join(plausible_values),
        "note": note,
    }


def build_weighted_descriptive_se(df, config):
    weight_col = config["pisa"]["student_weight"]
    replicate_cols = [
        col
        for col in replicate_weight_columns(
            config["pisa"]["replicate_weight_prefix"],
            config["pisa"]["replicate_weight_count"],
        )
        if col in df.columns
    ]
    pv_cols = [
        col
        for col in math_pv_columns(
            config["pisa"]["math_pv_count"],
            config["pisa"]["math_pv_prefix"],
            config["pisa"]["math_pv_suffix"],
        )
        if col in df.columns
    ]
    threshold = config["pisa"]["low_performer_threshold"]
    rows = []

    if pv_cols:
        estimates = []
        sampling_variances = []
        for pv_col in pv_cols:
            estimate, sampling_variance = weighted_replicate_estimate(
                df,
                df[pv_col],
                weight_col,
                replicate_cols,
            )
            estimates.append(estimate)
            sampling_variances.append(sampling_variance)
        pooled = combine_plausible_value_estimates(estimates, sampling_variances)
        rows.append(
            descriptive_row(
                "math_score_mean_pv_pooled",
                pooled.estimate,
                pooled.sampling_variance,
                pooled.imputation_variance,
                pooled.total_variance,
                "Pooled across mathematics plausible values with BRR replicate-weight sampling variance.",
                len(df),
                weight_col,
                replicate_cols,
                pv_cols,
            )
        )

        estimates = []
        sampling_variances = []
        for pv_col in pv_cols:
            flag = (df[pv_col] < threshold).astype(float)
            estimate, sampling_variance = weighted_replicate_estimate(
                df,
                flag,
                weight_col,
                replicate_cols,
            )
            estimates.append(estimate)
            sampling_variances.append(sampling_variance)
        pooled = combine_plausible_value_estimates(estimates, sampling_variances)
        rows.append(
            descriptive_row(
                "low_performer_rate_pv_pooled",
                pooled.estimate,
                pooled.sampling_variance,
                pooled.imputation_variance,
                pooled.total_variance,
                "Pooled rate using each mathematics plausible value against the Level 2 threshold.",
                len(df),
                weight_col,
                replicate_cols,
                pv_cols,
            )
        )

    for measure, value_col, note in [
        (
            "math_score_mean_model_outcome",
            "MATH_PV_MEAN",
            "Modeling outcome based on the row-wise mean of mathematics plausible values.",
        ),
        (
            "low_performer_rate_model_label",
            "LOW_PERFORMER_MATH",
            "Modeling label based on the row-wise mean of mathematics plausible values.",
        ),
    ]:
        if value_col not in df.columns:
            continue
        estimate, sampling_variance = weighted_replicate_estimate(
            df,
            df[value_col],
            weight_col,
            replicate_cols,
        )
        rows.append(
            descriptive_row(
                measure,
                estimate,
                sampling_variance,
                0.0,
                sampling_variance,
                note,
                len(df),
                weight_col,
                replicate_cols,
                pv_cols if value_col == "MATH_PV_MEAN" else [],
            )
        )

    return rows


def metadata_columns(path: Path) -> set[str]:
    if not path or not path.exists() or path.suffix.lower() != ".sav":
        return set()
    try:
        import pyreadstat

        _, meta = pyreadstat.read_sav(path, metadataonly=True)
        return set(meta.column_names)
    except Exception:
        return set()


def choose_first(paths: list[Path]) -> Path | None:
    return paths[0] if paths else None


def feature_source(feature: str, student_columns: set[str], school_columns: set[str]) -> str:
    in_student = feature in student_columns
    in_school = feature in school_columns
    if in_student and in_school:
        return "student questionnaire; school variable with same name also available"
    if in_student:
        return "student questionnaire"
    if in_school:
        return "school questionnaire"
    return "not found in raw questionnaire files"


def feature_decision(missing_rate: float | None, main_max: float, extreme_min: float) -> str:
    if missing_rate is None:
        return "unavailable"
    if missing_rate >= extreme_min:
        return "excluded_extreme_missingness"
    if missing_rate <= main_max:
        return "main_model"
    return "extended_or_robustness_only"


def add_escs_quintile(df):
    if "ESCS" not in df.columns:
        return df
    result = df.copy()
    result["ESCS_QUINTILE"] = "missing"
    valid = result["ESCS"].notna()
    result.loc[valid, "ESCS_QUINTILE"] = (
        result.loc[valid, "ESCS"].rank(method="first").pipe(
            lambda s: __import__("pandas").qcut(s, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        )
    )
    return result


def main() -> int:
    config = load_config()
    processed = resolve_project_path(config["paths"]["processed_dir"]) / "pisa2022_math_model_frame.parquet"
    raw_dir = resolve_project_path(config["paths"]["raw_dir"])
    tables_dir = resolve_project_path(config["paths"]["tables_dir"])
    processed_dir = resolve_project_path(config["paths"]["processed_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = load_table(processed)
    weight = config["pisa"]["student_weight"]
    country = config["pisa"]["country"]
    main_max = config["models"].get("main_missingness_max", 0.5)
    extreme_min = config["models"].get("extreme_missingness_min", 0.8)

    student_file = choose_first(find_matching_files(raw_dir, config["files"]["student_patterns"]))
    school_file = choose_first(find_matching_files(raw_dir, config["files"]["school_patterns"]))
    student_columns = metadata_columns(student_file) if student_file else set()
    school_columns = metadata_columns(school_file) if school_file else set()

    group_lookup = feature_group_lookup(config["features"])
    configured_features = flatten_feature_config(config["features"])
    available_features = [feature for feature in configured_features if feature in df.columns]
    missing_features = [feature for feature in configured_features if feature not in df.columns]
    missing_rates = df[available_features].isna().mean().to_dict()

    audit_rows = []
    for feature in configured_features:
        missing_rate = missing_rates.get(feature)
        audit_rows.append(
            {
                "feature": feature,
                "construct": FEATURE_CONSTRUCTS.get(feature, feature),
                "configured_group": group_lookup.get(feature, "unknown"),
                "source": feature_source(feature, student_columns, school_columns),
                "available_in_processed": feature in df.columns,
                "missing_rate": missing_rate,
                "decision": feature_decision(missing_rate, main_max, extreme_min),
            }
        )

    import pandas as pd

    variable_audit = pd.DataFrame(audit_rows)
    variable_audit.to_csv(tables_dir / "variable_audit.csv", index=False)

    main_features = variable_audit.loc[variable_audit["decision"] == "main_model", "feature"].tolist()
    extended_features = variable_audit.loc[
        variable_audit["decision"].isin(["main_model", "extended_or_robustness_only"]),
        "feature",
    ].tolist()
    feature_sets = {
        "main_features": main_features,
        "extended_features": extended_features,
        "excluded_extreme_missingness": variable_audit.loc[
            variable_audit["decision"] == "excluded_extreme_missingness",
            "feature",
        ].tolist(),
        "unavailable_features": missing_features,
        "main_missingness_max": main_max,
        "extreme_missingness_min": extreme_min,
    }
    (processed_dir / "feature_sets.json").write_text(
        json.dumps(feature_sets, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    country_rows = []
    for country_code, group in df.groupby(country, dropna=False, observed=False):
        country_rows.append(
            {
                "country": country_code,
                "n_unweighted": len(group),
                "weighted_math_mean": weighted_mean(group, "MATH_PV_MEAN", weight),
                "low_performer_rate_weighted": weighted_rate(group, "LOW_PERFORMER_MATH", weight),
                "low_performer_rate_unweighted": float(group["LOW_PERFORMER_MATH"].mean()),
            }
        )
    by_country = pd.DataFrame(country_rows).sort_values("country")
    by_country.to_csv(tables_dir / "sample_descriptives_by_country.csv", index=False)

    sample_summary = pd.DataFrame(
        [
            {
                "n_students": len(df),
                "n_countries": df[country].nunique(dropna=True),
                "weighted_math_mean": weighted_mean(df, "MATH_PV_MEAN", weight),
                "low_performer_rate_weighted": weighted_rate(df, "LOW_PERFORMER_MATH", weight),
                "low_performer_rate_unweighted": float(df["LOW_PERFORMER_MATH"].mean()),
                "available_configured_features": len(available_features),
                "main_model_features": len(main_features),
                "extended_features": len(extended_features),
                "student_file": str(student_file) if student_file else "",
                "school_file": str(school_file) if school_file else "",
                "school_features_used": ", ".join(
                    variable_audit.loc[
                        (variable_audit["available_in_processed"])
                        & (variable_audit["source"] == "school questionnaire"),
                        "feature",
                    ].tolist()
                ),
            }
        ]
    )
    sample_summary.to_csv(tables_dir / "sample_summary.csv", index=False)
    pd.DataFrame(build_weighted_descriptive_se(df, config)).to_csv(
        tables_dir / "weighted_descriptive_se.csv",
        index=False,
    )

    subgroup_rows = []
    grouped_df = add_escs_quintile(df)
    for variable in ["ST004D01T", "IMMIG", "ESCS_QUINTILE"]:
        if variable not in grouped_df.columns:
            continue
        for value, group in grouped_df.groupby(variable, dropna=False, observed=False):
            subgroup_rows.append(
                {
                    "group_variable": variable,
                    "group_value": value,
                    "n_unweighted": len(group),
                    "weighted_math_mean": weighted_mean(group, "MATH_PV_MEAN", weight),
                    "low_performer_rate_weighted": weighted_rate(
                        group, "LOW_PERFORMER_MATH", weight
                    ),
                }
            )
    pd.DataFrame(subgroup_rows).to_csv(tables_dir / "subgroup_descriptives.csv", index=False)

    print(sample_summary.to_string(index=False))
    print("\nVariable decisions:")
    print(variable_audit["decision"].value_counts(dropna=False).to_string())
    print("\nCountry descriptives preview:")
    print(by_country.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
