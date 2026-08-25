#!/usr/bin/env python3
"""Institution-level cold-start validation on the PISA 2022 model frame.

This is a v5 candidate branch. It holds out complete schools within each
country/economy, fits one frozen XGBoost model per plausible value, and evaluates
the unseen-school holdout with all 80 Fay--BRR replicate weights. It is not an
external institution or deployment study.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import pyreadstat
import sklearn
import xgboost

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pisa_xai.config import load_config, resolve_project_path
from pisa_xai.io import load_table
from pisa_xai.pisa import math_pv_columns, replicate_weight_columns
from pisa_xai.v5_survey import (
    ensure_v5_output_path,
    fay_brr_variance,
    institutional_cold_start_split_indices,
    pool_pv_estimates,
    validate_weights,
)


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260510
THRESHOLD = 420.07


def load_core_module():
    path = ROOT / "scripts" / "33_pisa_pv_replicate_weight_audit.py"
    spec = importlib.util.spec_from_file_location("v5_route_a_core", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load core Route A module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a non-publication smoke path.")
    parser.add_argument("--smoke-rows", type=int, default=30_000)
    parser.add_argument("--n-jobs", type=int, default=10)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    return parser.parse_args()


def write_csv(frame: pd.DataFrame, path: Path, required: list[str]) -> None:
    ensure_v5_output_path(path)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"output schema missing: {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> int:
    args = parse_args()
    if args.n_jobs < 1:
        raise ValueError("n-jobs must be positive")
    started = perf_counter()
    core = load_core_module()
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
    for path in (raw_path, frame_path, feature_path, classifier_path, regressor_path):
        if not path.exists():
            raise FileNotFoundError(path)

    frame = load_table(frame_path)
    frame = core.read_senate_weights(raw_path, frame)
    pv_columns = math_pv_columns(10)
    replicate_columns = replicate_weight_columns("W_FSTURWT", 80)
    required = ["CNT", "CNTSCHID", "W_FSTUWT", "SENWT", *pv_columns, *replicate_columns]
    for column in required:
        if column not in frame.columns:
            raise ValueError(f"required field is missing: {column}")
    for column in ["W_FSTUWT", "SENWT", *replicate_columns]:
        validate_weights(frame[column])
    if frame[pv_columns].isna().any().any():
        raise ValueError("one or more plausible values are missing")

    if args.smoke:
        if args.smoke_rows >= len(frame):
            raise ValueError("smoke rows must be smaller than the full frame")
        frame = frame.sample(args.smoke_rows, random_state=SEED).sort_index().copy()

    feature_sets = json.loads(feature_path.read_text(encoding="utf-8"))
    features = [name for name in feature_sets["main_features"] if name in frame.columns]
    if len(features) != 33:
        raise ValueError(f"expected 33 frozen features, found {len(features)}")
    matrix = core.make_feature_matrix(frame, features)
    train_index, test_index, split_audit = institutional_cold_start_split_indices(
        frame,
        test_fraction=args.test_fraction,
        random_state=SEED,
    )
    train_frame = frame.loc[train_index]
    test_frame = frame.loc[test_index]
    train_schools = set(zip(train_frame["CNT"].astype(str), train_frame["CNTSCHID"]))
    test_schools = set(zip(test_frame["CNT"].astype(str), test_frame["CNTSCHID"]))
    overlap = train_schools.intersection(test_schools)
    if overlap:
        raise RuntimeError(f"school leakage detected: {len(overlap)} overlapping schools")
    if split_audit["schools_train"].lt(1).any() or split_audit["schools_test"].lt(1).any():
        raise RuntimeError("a country has no train or test school")
    x_train = matrix.loc[train_index]
    x_test = matrix.loc[test_index]
    train_weight = core.normalized_weight(train_frame["W_FSTUWT"])
    test_weight = core.normalized_weight(test_frame["W_FSTUWT"])
    senate_weight = core.normalized_weight(test_frame["SENWT"])
    masks = core.make_group_masks(test_frame, minimum_group_size=20 if args.smoke else 200)
    group_counts = {name: int(mask.sum()) for name, mask in masks.items()}
    group_counts["contrast"] = group_counts["low_ses_non_native"] + group_counts["high_ses_native"]

    classifier_base = joblib.load(classifier_path)
    regressor_base = joblib.load(regressor_path)
    specific_rows: list[dict[str, object]] = []
    replicate_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    full_population: dict[tuple[str, str, str, str], float] = {}
    full_senate: dict[tuple[str, str, str, str], float] = {}
    replicate_values: dict[tuple[str, str, str, str], list[float]] = {}

    for pv in pv_columns:
        print(f"Training/evaluating unseen-school split for {pv}", flush=True)
        y_reg_train = train_frame[pv].to_numpy(dtype=float)
        y_reg_test = test_frame[pv].to_numpy(dtype=float)
        y_clf_train = (y_reg_train < THRESHOLD).astype(int)
        y_clf_test = (y_reg_test < THRESHOLD).astype(int)
        if np.unique(y_clf_train).size != 2 or np.unique(y_clf_test).size != 2:
            raise ValueError(f"{pv} classification target lacks both classes in train/test")
        regressor = core.build_model(regressor_base, args.n_jobs)
        classifier = core.build_model(classifier_base, args.n_jobs)
        regressor.fit(x_train, y_reg_train, sample_weight=train_weight)
        classifier.fit(x_train, y_clf_train, sample_weight=train_weight)
        reg_prediction = regressor.predict(x_test)
        clf_probability = classifier.predict_proba(x_test)[:, 1]
        predictions.append(
            pd.DataFrame(
                {
                    "row_index": test_frame.index,
                    "pv": pv,
                    "country": test_frame["CNT"].astype(str).to_numpy(),
                    "school_id": test_frame["CNTSCHID"].to_numpy(),
                    "y_regression": y_reg_test,
                    "y_classification": y_clf_test,
                    "regression_prediction": reg_prediction,
                    "classification_probability": clf_probability,
                }
            )
        )
        population_metrics = core.calculate_metrics(
            y_reg_test, reg_prediction, y_clf_test, clf_probability, test_weight, masks
        )
        senate_metrics = core.calculate_metrics(
            y_reg_test, reg_prediction, y_clf_test, clf_probability, senate_weight, masks
        )
        specific_rows.extend(core.metric_records(pv, "population", population_metrics, group_counts))
        specific_rows.extend(core.metric_records(pv, "senate", senate_metrics, group_counts))
        for key, value in population_metrics.items():
            full_population[(pv, *key)] = value
        for key, value in senate_metrics.items():
            full_senate[(pv, *key)] = value
        for replicate_number, column in enumerate(replicate_columns, start=1):
            replicate_weight = core.normalized_weight(test_frame[column])
            try:
                values = core.calculate_metrics(
                    y_reg_test, reg_prediction, y_clf_test, clf_probability, replicate_weight, masks
                )
            except (core.MetricNotComputable, ValueError) as exc:
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
            for (task, group, metric), estimate in values.items():
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
    for (pv, task, group, metric), full_estimate in full_population.items():
        reps = replicate_values.get((pv, task, group, metric), [])
        failed = 80 - len(reps)
        variance = fay_brr_variance(full_estimate, reps) if failed == 0 else np.nan
        if failed > 4:
            failures.append(
                {
                    "phase": "replicate_variance",
                    "pv": pv,
                    "replicate": "",
                    "component": f"{task}/{group}/{metric}",
                    "reason": f"{failed} of 80 replicate estimates were invalid",
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
                "valid_replicates": len(reps),
                "failed_replicates": failed,
                "sampling_variance": variance,
                "sampling_standard_error": np.sqrt(variance) if np.isfinite(variance) else np.nan,
            }
        )

    summary = pd.DataFrame(uncertainty_rows)
    pooled_rows: list[dict[str, object]] = []
    senate_rows: list[dict[str, object]] = []
    all_keys = sorted({(task, group, metric) for _, task, group, metric in full_population})
    for task, group, metric in all_keys:
        estimates = [full_population[(pv, task, group, metric)] for pv in pv_columns]
        subset = summary[(summary.task == task) & (summary.group == group) & (summary.metric == metric)].set_index("pv")
        variances = [subset.loc[pv, "sampling_variance"] for pv in pv_columns]
        if np.isfinite(variances).all():
            pooled = dict(pool_pv_estimates(estimates, variances))
            status = "INSTITUTION_COLD_START_CANDIDATE"
            lower = pooled["estimate"] - 1.96 * pooled["standard_error"]
            upper = pooled["estimate"] + 1.96 * pooled["standard_error"]
        else:
            pooled = {
                "estimate": float(np.mean(estimates)),
                "sampling_variance": np.nan,
                "imputation_variance": float(np.var(estimates, ddof=1)),
                "total_variance": np.nan,
                "standard_error": np.nan,
            }
            status = "INVALID_REPLICATE_VARIANCE"
            lower = upper = np.nan
        pooled_rows.append(
            {
                "estimand": "population",
                "task": task,
                "group": group,
                "metric": metric,
                "pv_count": 10,
                **pooled,
                "ci_lower": lower,
                "ci_upper": upper,
                "status": status,
            }
        )
        senate_rows.append(
            {
                "task": task,
                "group": group,
                "metric": metric,
                "population_pv_pooled_estimate": float(np.mean(estimates)),
                "senate_pv_pooled_estimate": float(np.mean([full_senate[(pv, task, group, metric)] for pv in pv_columns])),
                "senate_minus_population": float(np.mean([full_senate[(pv, task, group, metric)] for pv in pv_columns]) - np.mean(estimates)),
                "senate_variance_status": "POINT_ESTIMAND_SENSITIVITY_ONLY",
            }
        )

    specific = pd.DataFrame(specific_rows)
    pooled = pd.DataFrame(pooled_rows)
    senate = pd.DataFrame(senate_rows)
    intersection = pooled[
        (pooled.task == "classification")
        & (pooled.group == "contrast")
        & (pooled.metric.isin(["auc_difference", "ece_difference", "slope_difference"]))
    ].copy()
    failures_frame = pd.DataFrame(failures, columns=["phase", "pv", "replicate", "component", "reason"])
    output_dir = interim_dir / "v5_institution_cold_start_smoke" if args.smoke else tables_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "pv_specific_metrics": output_dir / "v5_institution_cold_start_pv_specific_metrics.csv",
        "replicate_uncertainty": output_dir / "v5_institution_cold_start_replicate_uncertainty.csv",
        "pooled_metrics": output_dir / "v5_institution_cold_start_pooled_metrics.csv",
        "senate_sensitivity": output_dir / "v5_institution_cold_start_senate_sensitivity.csv",
        "intersectional_ci": output_dir / "v5_institution_cold_start_intersectional_ci.csv",
        "failures": output_dir / "v5_institution_cold_start_failures.csv",
        "split_audit": output_dir / "v5_institution_cold_start_split_audit.csv",
        "manifest": output_dir / "v5_institution_cold_start_manifest.json",
        "predictions": interim_dir / "v5_institution_cold_start_predictions.parquet",
    }
    write_csv(specific, outputs["pv_specific_metrics"], ["pv", "estimand", "task", "group", "metric", "estimate", "n"])
    write_csv(pd.concat([pd.DataFrame(replicate_rows), summary], ignore_index=True, sort=False), outputs["replicate_uncertainty"], ["row_type", "pv", "task", "group", "metric"])
    write_csv(pooled, outputs["pooled_metrics"], ["estimand", "task", "group", "metric", "estimate", "status"])
    write_csv(senate, outputs["senate_sensitivity"], ["task", "group", "metric", "population_pv_pooled_estimate", "senate_pv_pooled_estimate"])
    write_csv(intersection, outputs["intersectional_ci"], ["metric", "estimate", "ci_lower", "ci_upper", "status"])
    write_csv(failures_frame, outputs["failures"], ["phase", "pv", "replicate", "component", "reason"])
    write_csv(split_audit, outputs["split_audit"], ["country", "schools_total", "schools_train", "schools_test"])
    outputs["predictions"].parent.mkdir(parents=True, exist_ok=True)
    pd.concat(predictions, ignore_index=True).to_parquet(outputs["predictions"], index=False)
    manifest = {
        "status": "SMOKE_DEBUG_ONLY" if args.smoke else "FULL_DATA_INSTITUTION_COLD_START_CANDIDATE",
        "seed": SEED,
        "test_fraction": args.test_fraction,
        "institution_unit": "CNT-CNTSCHID",
        "split_rule": "within-country deterministic school split; no school overlap",
        "full_rows": int(len(frame)),
        "n_train": int(len(train_frame)),
        "n_holdout": int(len(test_frame)),
        "n_train_schools": int(len(train_schools)),
        "n_holdout_schools": int(len(test_schools)),
        "n_countries": int(split_audit["country"].nunique()),
        "features": features,
        "pv_columns": pv_columns,
        "replicate_weight_columns": replicate_columns,
        "group_counts": group_counts,
        "school_overlap_count": len(overlap),
        "failure_count": int(len(failures_frame)),
        "model": {
            "classifier_base": str(classifier_path),
            "regressor_base": str(regressor_path),
            "sample_weight": "normalized W_FSTUWT by train/test frame separately",
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
        "outputs": {name: str(path) for name, path in outputs.items()},
        "inference_boundary": "unseen-school fixed-model Fay-BRR evaluation; not external institution validation, full training uncertainty, or individual inference",
        "elapsed_seconds": round(perf_counter() - started, 2),
    }
    ensure_v5_output_path(outputs["manifest"])
    outputs["manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    return 0 if len(failures_frame) <= 4 else 2


if __name__ == "__main__":
    raise SystemExit(main())

