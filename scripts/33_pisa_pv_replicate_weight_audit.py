#!/usr/bin/env python3
"""Route A PISA model-level PV and replicate-weight validation.

This script deliberately writes only v5-prefixed candidate artifacts. It uses
the legacy split and hyperparameters, but trains one model per plausible value
with full student weights and reports uncertainty conditional on fitted models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd
import pyreadstat
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score

from pisa_xai.config import load_config, resolve_project_path
from pisa_xai.io import load_table
from pisa_xai.pisa import math_pv_columns, replicate_weight_columns
from pisa_xai.v5_survey import (
    MetricNotComputable,
    ensure_v5_output_path,
    fay_brr_variance,
    fixed_legacy_split_indices,
    pool_pv_estimates,
    require_output_columns,
    validate_join,
    validate_weights,
    weighted_binary_metrics,
)


SEED = 20260510
THRESHOLD = 420.07
CORE_CONTRASTS = {
    ("classification", "contrast", "auc_difference"),
    ("classification", "contrast", "ece_difference"),
    ("classification", "contrast", "slope_difference"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_weight(values: pd.Series) -> np.ndarray:
    raw = validate_weights(values)
    return raw / raw.mean()


def read_senate_weights(raw_path: Path, frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["CNT", "CNTSTUID", "CNTSCHID", "W_FSTUWT", "SENWT"]
    raw, _ = pyreadstat.read_sav(raw_path, usecols=columns)
    for data in (raw, frame):
        data["CNT"] = data["CNT"].astype(str).str.strip()
        data["CNTSTUID"] = pd.to_numeric(data["CNTSTUID"], errors="raise")
        data["CNTSCHID"] = pd.to_numeric(data["CNTSCHID"], errors="raise")
        data["W_FSTUWT"] = pd.to_numeric(data["W_FSTUWT"], errors="raise")
    return validate_join(
        frame,
        raw.drop(columns=["CNT"]),
        keys=["CNTSTUID"],
        matching_columns=["CNTSCHID", "W_FSTUWT"],
    )


def make_feature_matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    matrix = frame[features].copy()
    for column in matrix.columns:
        if str(matrix[column].dtype) == "category":
            matrix[column] = matrix[column].cat.codes.astype("float32")
    return matrix.astype("float32")


def make_group_masks(frame: pd.DataFrame, minimum_group_size: int = 200) -> dict[str, np.ndarray]:
    escs = frame["ESCS"]
    quartile = pd.Series("missing", index=frame.index, dtype="object")
    valid = escs.notna()
    quartile.loc[valid] = pd.qcut(
        escs.loc[valid].rank(method="first"),
        4,
        labels=["Q1", "Q2", "Q3", "Q4"],
    ).astype(str)
    immigration = frame["IMMIG"].astype("string").str.strip().str.lower()
    non_native = immigration.str.startswith(("first", "second"), na=False).to_numpy()
    native = immigration.str.startswith("native", na=False).to_numpy()
    low_intersection = ((quartile == "Q1").to_numpy()) & non_native
    high_reference = ((quartile == "Q4").to_numpy()) & native
    if low_intersection.sum() < minimum_group_size or high_reference.sum() < minimum_group_size:
        raise ValueError(
            f"C1 subgroup minimum of {minimum_group_size} observations is not met"
        )
    return {
        "global": np.ones(len(frame), dtype=bool),
        "low_ses_non_native": low_intersection,
        "high_ses_native": high_reference,
    }


def weighted_regression_metrics(
    y_true: np.ndarray, prediction: np.ndarray, weight: np.ndarray
) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, prediction, sample_weight=weight))),
        "r2": float(r2_score(y_true, prediction, sample_weight=weight)),
    }


def calculate_metrics(
    y_regression: np.ndarray,
    regression_prediction: np.ndarray,
    y_classification: np.ndarray,
    classification_probability: np.ndarray,
    weight: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[tuple[str, str, str], float]:
    values: dict[tuple[str, str, str], float] = {}
    for metric, estimate in weighted_regression_metrics(
        y_regression, regression_prediction, weight
    ).items():
        values[("regression", "global", metric)] = estimate
    class_by_group: dict[str, dict[str, float]] = {}
    for group, mask in masks.items():
        class_by_group[group] = dict(
            weighted_binary_metrics(
                y_classification[mask], classification_probability[mask], weight[mask]
            )
        )
        for metric, estimate in class_by_group[group].items():
            values[("classification", group, metric)] = estimate
    values[("classification", "contrast", "auc_difference")] = (
        class_by_group["low_ses_non_native"]["auc"]
        - class_by_group["high_ses_native"]["auc"]
    )
    values[("classification", "contrast", "ece_difference")] = (
        class_by_group["low_ses_non_native"]["ece"] - class_by_group["global"]["ece"]
    )
    values[("classification", "contrast", "slope_difference")] = (
        class_by_group["low_ses_non_native"]["calibration_slope"]
        - class_by_group["global"]["calibration_slope"]
    )
    return values


def build_model(base_model, n_jobs: int):
    return clone(base_model).set_params(
        n_jobs=n_jobs,
        random_state=SEED,
        verbosity=0,
    )


def metric_records(
    pv: str,
    estimand: str,
    values: dict[tuple[str, str, str], float],
    group_counts: dict[str, int],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for (task, group, metric), estimate in values.items():
        records.append(
            {
                "pv": pv,
                "estimand": estimand,
                "task": task,
                "group": group,
                "metric": metric,
                "estimate": estimate,
                "n": group_counts.get(group, np.nan),
            }
        )
    return records


def write_candidate_csv(frame: pd.DataFrame, path: Path, required: list[str]) -> None:
    ensure_v5_output_path(path)
    require_output_columns(frame, required)
    frame.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a non-publication path check.")
    parser.add_argument("--smoke-rows", type=int, default=20_000)
    parser.add_argument("--n-jobs", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = perf_counter()
    config = load_config()
    processed_dir = resolve_project_path(config["paths"]["processed_dir"])
    tables_dir = resolve_project_path(config["paths"]["tables_dir"])
    interim_dir = resolve_project_path(config["paths"]["interim_dir"])
    raw_dir = resolve_project_path(config["paths"]["raw_dir"])
    raw_path = raw_dir / "CY08MSP_STU_QQQ.SAV"
    frame_path = processed_dir / "pisa2022_math_model_frame.parquet"
    feature_path = processed_dir / "feature_sets.json"
    classifier_path = processed_dir / "models" / "classification_xgboost_tuned.joblib"
    regressor_path = processed_dir / "models" / "regression_xgboost_tuned.joblib"
    for path in [raw_path, frame_path, feature_path, classifier_path, regressor_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    frame = load_table(frame_path)
    frame = read_senate_weights(raw_path, frame)
    pv_columns = math_pv_columns(10)
    replicate_columns = replicate_weight_columns("W_FSTURWT", 80)
    for column in ["W_FSTUWT", "SENWT", *pv_columns, *replicate_columns]:
        if column not in frame.columns:
            raise ValueError(f"required field is missing: {column}")
        validate_weights(frame[column]) if column in {"W_FSTUWT", "SENWT", *replicate_columns} else None
    if frame[pv_columns].isna().any().any():
        raise ValueError("one or more plausible values are missing")

    if args.smoke:
        if args.smoke_rows >= len(frame):
            raise ValueError("smoke rows must be smaller than the full frame")
        legacy = frame["LOW_PERFORMER_MATH"]
        sampled, _ = __import__("sklearn.model_selection", fromlist=["train_test_split"]).train_test_split(
            frame,
            train_size=args.smoke_rows,
            stratify=legacy,
            random_state=SEED,
        )
        frame = sampled.sort_index().copy()

    feature_sets = json.loads(feature_path.read_text(encoding="utf-8"))
    features = [name for name in feature_sets["main_features"] if name in frame.columns]
    if len(features) != 33:
        raise ValueError(f"expected 33 frozen features, found {len(features)}")
    matrix = make_feature_matrix(frame, features)
    train_index, test_index = fixed_legacy_split_indices(
        frame.index, frame["LOW_PERFORMER_MATH"]
    )
    train_frame = frame.loc[train_index]
    test_frame = frame.loc[test_index]
    x_train = matrix.loc[train_index]
    x_test = matrix.loc[test_index]
    population_train_weight = normalized_weight(train_frame["W_FSTUWT"])
    population_test_weight = normalized_weight(test_frame["W_FSTUWT"])
    senate_test_weight = normalized_weight(test_frame["SENWT"])
    masks = make_group_masks(test_frame, minimum_group_size=20 if args.smoke else 200)
    group_counts = {name: int(mask.sum()) for name, mask in masks.items()}
    group_counts["contrast"] = group_counts["low_ses_non_native"] + group_counts["high_ses_native"]

    classifier_base = joblib.load(classifier_path)
    regressor_base = joblib.load(regressor_path)
    specific_rows: list[dict[str, object]] = []
    replicate_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    full_population: dict[tuple[str, str, str, str], float] = {}
    full_senate: dict[tuple[str, str, str, str], float] = {}
    replicate_values: dict[tuple[str, str, str, str], list[float]] = {}

    for pv in pv_columns:
        print(f"Training/evaluating {pv}", flush=True)
        y_reg_train = train_frame[pv].to_numpy(dtype=float)
        y_reg_test = test_frame[pv].to_numpy(dtype=float)
        y_clf_train = (y_reg_train < THRESHOLD).astype(int)
        y_clf_test = (y_reg_test < THRESHOLD).astype(int)
        if np.unique(y_clf_train).size != 2 or np.unique(y_clf_test).size != 2:
            raise ValueError(f"{pv} does not contain both classes in the fixed split")

        regressor = build_model(regressor_base, args.n_jobs)
        classifier = build_model(classifier_base, args.n_jobs)
        regressor.fit(x_train, y_reg_train, sample_weight=population_train_weight)
        classifier.fit(x_train, y_clf_train, sample_weight=population_train_weight)
        reg_prediction = regressor.predict(x_test)
        clf_probability = classifier.predict_proba(x_test)[:, 1]

        prediction_frames.append(
            pd.DataFrame(
                {
                    "row_index": test_frame.index,
                    "pv": pv,
                    "y_regression": y_reg_test,
                    "y_classification": y_clf_test,
                    "regression_prediction": reg_prediction,
                    "classification_probability": clf_probability,
                }
            )
        )
        try:
            population_metrics = calculate_metrics(
                y_reg_test,
                reg_prediction,
                y_clf_test,
                clf_probability,
                population_test_weight,
                masks,
            )
        except (MetricNotComputable, ValueError) as exc:
            raise RuntimeError(f"full-sample population metrics failed for {pv}: {exc}") from exc
        senate_metrics = calculate_metrics(
            y_reg_test,
            reg_prediction,
            y_clf_test,
            clf_probability,
            senate_test_weight,
            masks,
        )
        specific_rows.extend(metric_records(pv, "population", population_metrics, group_counts))
        specific_rows.extend(metric_records(pv, "senate", senate_metrics, group_counts))
        for (task, group, metric), estimate in population_metrics.items():
            full_population[(pv, task, group, metric)] = estimate
        for (task, group, metric), estimate in senate_metrics.items():
            full_senate[(pv, task, group, metric)] = estimate

        for replicate_number, column in enumerate(replicate_columns, start=1):
            replicate_weight = normalized_weight(test_frame[column])
            try:
                replicate_metrics = calculate_metrics(
                    y_reg_test,
                    reg_prediction,
                    y_clf_test,
                    clf_probability,
                    replicate_weight,
                    masks,
                )
            except (MetricNotComputable, ValueError) as exc:
                failures.append(
                    {
                        "phase": "replicate_metric",
                        "pv": pv,
                        "replicate": replicate_number,
                        "component": "all",
                        "reason": str(exc),
                    }
                )
                continue
            for (task, group, metric), estimate in replicate_metrics.items():
                key = (pv, task, group, metric)
                replicate_values.setdefault(key, []).append(estimate)
                replicate_rows.append(
                    {
                        "row_type": "replicate",
                        "pv": pv,
                        "task": task,
                        "group": group,
                        "metric": metric,
                        "replicate": replicate_number,
                        "estimate": estimate,
                    }
                )

    uncertainty_rows: list[dict[str, object]] = []
    for key, full_estimate in full_population.items():
        pv, task, group, metric = key
        replicate_estimates = replicate_values.get(key, [])
        failed_count = 80 - len(replicate_estimates)
        variance = np.nan
        if failed_count == 0:
            variance = fay_brr_variance(full_estimate, replicate_estimates)
        elif failed_count > 4:
            failures.append(
                {
                    "phase": "replicate_variance",
                    "pv": pv,
                    "replicate": "",
                    "component": f"{task}/{group}/{metric}",
                    "reason": f"{failed_count} of 80 replicate estimates were invalid",
                }
            )
        uncertainty_rows.append(
            {
                "row_type": "summary",
                "pv": pv,
                "task": task,
                "group": group,
                "metric": metric,
                "full_estimate": full_estimate,
                "valid_replicates": len(replicate_estimates),
                "failed_replicates": failed_count,
                "sampling_variance": variance,
                "sampling_standard_error": np.sqrt(variance) if np.isfinite(variance) else np.nan,
            }
        )
    replicate_frame = pd.concat(
        [pd.DataFrame(replicate_rows), pd.DataFrame(uncertainty_rows)],
        ignore_index=True,
        sort=False,
    )

    pooled_rows: list[dict[str, object]] = []
    population_senate_rows: list[dict[str, object]] = []
    summary_frame = pd.DataFrame(uncertainty_rows)
    all_metric_keys = sorted({key[1:] for key in full_population})
    for task, group, metric in all_metric_keys:
        population_estimates = [
            full_population[(pv, task, group, metric)] for pv in pv_columns
        ]
        subset = summary_frame[
            (summary_frame["task"] == task)
            & (summary_frame["group"] == group)
            & (summary_frame["metric"] == metric)
        ].set_index("pv")
        variance_values = [subset.loc[pv, "sampling_variance"] for pv in pv_columns]
        if np.isfinite(variance_values).all():
            pooled = dict(pool_pv_estimates(population_estimates, variance_values))
            status = "MANUSCRIPT_CANDIDATE_PENDING_STOP_CHECK"
        else:
            pooled = {
                "estimate": float(np.mean(population_estimates)),
                "sampling_variance": np.nan,
                "imputation_variance": float(np.var(population_estimates, ddof=1)),
                "total_variance": np.nan,
                "standard_error": np.nan,
            }
            status = "INVALID_REPLICATE_VARIANCE"
        pooled_rows.append(
            {
                "estimand": "population",
                "task": task,
                "group": group,
                "metric": metric,
                "pv_count": 10,
                **pooled,
                "ci_lower": pooled["estimate"] - 1.96 * pooled["standard_error"]
                if np.isfinite(pooled["standard_error"])
                else np.nan,
                "ci_upper": pooled["estimate"] + 1.96 * pooled["standard_error"]
                if np.isfinite(pooled["standard_error"])
                else np.nan,
                "status": status,
            }
        )
        senate_estimates = [full_senate[(pv, task, group, metric)] for pv in pv_columns]
        population_senate_rows.append(
            {
                "task": task,
                "group": group,
                "metric": metric,
                "population_pv_pooled_estimate": float(np.mean(population_estimates)),
                "senate_pv_pooled_estimate": float(np.mean(senate_estimates)),
                "senate_minus_population": float(np.mean(senate_estimates) - np.mean(population_estimates)),
                "senate_variance_status": "POINT_ESTIMAND_SENSITIVITY_ONLY",
            }
        )

    specific_frame = pd.DataFrame(specific_rows)
    pooled_frame = pd.DataFrame(pooled_rows)
    senate_frame = pd.DataFrame(population_senate_rows)
    intersection_frame = pooled_frame[
        (pooled_frame["task"] == "classification")
        & (pooled_frame["group"] == "contrast")
        & (pooled_frame["metric"].isin(["auc_difference", "ece_difference", "slope_difference"]))
    ].copy()
    failures_frame = pd.DataFrame(
        failures,
        columns=["phase", "pv", "replicate", "component", "reason"],
    )

    if args.smoke:
        output_dir = interim_dir / "v5_smoke"
        output_dir.mkdir(parents=True, exist_ok=True)
        status = "SMOKE_DEBUG_ONLY"
    else:
        output_dir = tables_dir
        status = "FULL_DATA_CANDIDATE"
    outputs = {
        "pv_specific_metrics": output_dir / "v5_pv_specific_metrics.csv",
        "replicate_weight_uncertainty": output_dir / "v5_replicate_weight_uncertainty.csv",
        "pv_pooled_metrics": output_dir / "v5_pv_pooled_metrics.csv",
        "population_vs_senate_weights": output_dir / "v5_population_vs_senate_weights.csv",
        "intersectional_design_aware_ci": output_dir / "v5_intersectional_design_aware_ci.csv",
        "analysis_failures": output_dir / "v5_analysis_failures.csv",
        "analysis_manifest": output_dir / "v5_analysis_manifest.json",
        "holdout_predictions": (interim_dir / "v5_pv_specific_holdout_predictions.parquet"),
    }
    write_candidate_csv(
        specific_frame,
        outputs["pv_specific_metrics"],
        ["pv", "estimand", "task", "group", "metric", "estimate", "n"],
    )
    write_candidate_csv(
        replicate_frame,
        outputs["replicate_weight_uncertainty"],
        ["row_type", "pv", "task", "group", "metric"],
    )
    write_candidate_csv(
        pooled_frame,
        outputs["pv_pooled_metrics"],
        ["estimand", "task", "group", "metric", "estimate", "status"],
    )
    write_candidate_csv(
        senate_frame,
        outputs["population_vs_senate_weights"],
        ["task", "group", "metric", "population_pv_pooled_estimate", "senate_pv_pooled_estimate"],
    )
    write_candidate_csv(
        intersection_frame,
        outputs["intersectional_design_aware_ci"],
        ["metric", "estimate", "ci_lower", "ci_upper", "status"],
    )
    write_candidate_csv(
        failures_frame,
        outputs["analysis_failures"],
        ["phase", "pv", "replicate", "component", "reason"],
    )
    prediction_path = outputs["holdout_predictions"]
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(prediction_path, index=False)

    manifest = {
        "status": status,
        "seed": SEED,
        "threshold": THRESHOLD,
        "full_rows": int(len(frame)),
        "n_train": int(len(train_frame)),
        "n_holdout": int(len(test_frame)),
        "features": features,
        "pv_columns": pv_columns,
        "replicate_weight_columns": replicate_columns,
        "model": {
            "classifier_base": str(classifier_path),
            "regressor_base": str(regressor_path),
            "classifier_params": classifier_base.get_params(),
            "regressor_params": regressor_base.get_params(),
            "sample_weight": "W_FSTUWT normalized by train-frame mean",
        },
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pyreadstat": pyreadstat.__version__,
        },
        "source_hashes": {
            "model_frame": sha256(frame_path),
            "feature_sets": sha256(feature_path),
            "classifier_base": sha256(classifier_path),
            "regressor_base": sha256(regressor_path),
        },
        "group_counts": group_counts,
        "failure_count": int(len(failures_frame)),
        "elapsed_seconds": round(perf_counter() - started, 2),
        "outputs": {name: str(path) for name, path in outputs.items()},
        "inference_boundary": "fixed-model evaluation uncertainty from Fay-BRR; not full training uncertainty or individual inference",
    }
    ensure_v5_output_path(outputs["analysis_manifest"])
    outputs["analysis_manifest"].write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
