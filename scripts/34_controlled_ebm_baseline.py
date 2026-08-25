#!/usr/bin/env python3
"""Full-data, same-split, same-weight additive EBM baseline for Route A."""
from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

import interpret

from pisa_xai.config import load_config, resolve_project_path
from pisa_xai.io import load_table
from pisa_xai.pisa import math_pv_columns, replicate_weight_columns
from pisa_xai.v5_survey import (
    controlled_ebm_config,
    ensure_v5_output_path,
    fay_brr_variance,
    fixed_legacy_split_indices,
    pool_pv_estimates,
    validate_weights,
    weighted_binary_metrics,
)


SEED = 20260510
THRESHOLD = 420.07


def normalized_weight(values: pd.Series) -> np.ndarray:
    weights = validate_weights(values)
    return weights / weights.mean()


def feature_matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    matrix = frame[features].copy()
    for column in matrix.columns:
        if str(matrix[column].dtype) == "category":
            matrix[column] = matrix[column].cat.codes.astype("float32")
    return matrix.astype("float32")


def regression_metrics(y_true: np.ndarray, prediction: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, prediction, sample_weight=weight))),
        "r2": float(r2_score(y_true, prediction, sample_weight=weight)),
    }


def rss_value() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Non-publication EBM path check.")
    parser.add_argument("--smoke-rows", type=int, default=20_000)
    parser.add_argument("--n-jobs", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = perf_counter()
    config = load_config()
    processed_dir = resolve_project_path(config["paths"]["processed_dir"])
    tables_dir = resolve_project_path(config["paths"]["tables_dir"])
    interim_dir = resolve_project_path(config["paths"]["interim_dir"])
    frame = load_table(processed_dir / "pisa2022_math_model_frame.parquet")
    if args.smoke:
        if args.smoke_rows >= len(frame):
            raise ValueError("smoke rows must be smaller than the full frame")
        from sklearn.model_selection import train_test_split

        frame, _ = train_test_split(
            frame,
            train_size=args.smoke_rows,
            stratify=frame["LOW_PERFORMER_MATH"],
            random_state=SEED,
        )
        frame = frame.sort_index().copy()
        output_path = interim_dir / "v5_ebm_smoke" / "v5_controlled_ebm_baseline.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        status = "SMOKE_DEBUG_ONLY"
    else:
        predecessor = tables_dir / "v5_analysis_manifest.json"
        if not predecessor.exists():
            raise FileNotFoundError("run full script 33 before the controlled EBM baseline")
        predecessor_status = json.loads(predecessor.read_text(encoding="utf-8")).get("status")
        if predecessor_status != "FULL_DATA_CANDIDATE":
            raise RuntimeError("the preceding v5 XGBoost analysis is not a full-data candidate run")
        output_path = tables_dir / "v5_controlled_ebm_baseline.csv"
        status = "FULL_DATA_CANDIDATE"

    features = json.loads((processed_dir / "feature_sets.json").read_text(encoding="utf-8"))["main_features"]
    features = [feature for feature in features if feature in frame.columns]
    if len(features) != 33:
        raise ValueError(f"expected 33 frozen features, found {len(features)}")
    matrix = feature_matrix(frame, features)
    train_index, test_index = fixed_legacy_split_indices(frame.index, frame["LOW_PERFORMER_MATH"])
    train_frame = frame.loc[train_index]
    test_frame = frame.loc[test_index]
    x_train = matrix.loc[train_index]
    x_test = matrix.loc[test_index]
    train_weight = normalized_weight(train_frame["W_FSTUWT"])
    test_weight = normalized_weight(test_frame["W_FSTUWT"])
    replicate_columns = replicate_weight_columns("W_FSTURWT", 80)
    for column in replicate_columns:
        validate_weights(test_frame[column])

    model_config = dict(controlled_ebm_config(args.n_jobs))
    rows: list[dict[str, object]] = []
    pooled_inputs: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for pv in math_pv_columns(10):
        print(f"Training controlled EBM for {pv}", flush=True)
        y_reg_train = train_frame[pv].to_numpy(dtype=float)
        y_reg_test = test_frame[pv].to_numpy(dtype=float)
        y_clf_train = (y_reg_train < THRESHOLD).astype(int)
        y_clf_test = (y_reg_test < THRESHOLD).astype(int)
        before_rss = rss_value()
        classifier_start = perf_counter()
        classifier = ExplainableBoostingClassifier(**model_config)
        classifier.fit(x_train, y_clf_train, sample_weight=train_weight)
        classifier_seconds = perf_counter() - classifier_start
        probability = classifier.predict_proba(x_test)[:, 1]
        classifier_metrics = dict(weighted_binary_metrics(y_clf_test, probability, test_weight))

        regressor_start = perf_counter()
        regressor = ExplainableBoostingRegressor(**model_config)
        regressor.fit(x_train, y_reg_train, sample_weight=train_weight)
        regressor_seconds = perf_counter() - regressor_start
        prediction = regressor.predict(x_test)
        regressor_metrics = regression_metrics(y_reg_test, prediction, test_weight)
        after_rss = rss_value()

        full_metrics = {
            **{("classification", metric): value for metric, value in classifier_metrics.items()},
            **{("regression", metric): value for metric, value in regressor_metrics.items()},
        }
        replicate_values: dict[tuple[str, str], list[float]] = {key: [] for key in full_metrics}
        for column in replicate_columns:
            replicate_weight = normalized_weight(test_frame[column])
            values = {
                **{
                    ("classification", metric): value
                    for metric, value in weighted_binary_metrics(
                        y_clf_test, probability, replicate_weight
                    ).items()
                },
                **{
                    ("regression", metric): value
                    for metric, value in regression_metrics(
                        y_reg_test, prediction, replicate_weight
                    ).items()
                },
            }
            for key, value in values.items():
                replicate_values[key].append(value)
        for (task, metric), value in full_metrics.items():
            variance = fay_brr_variance(value, replicate_values[(task, metric)])
            pooled_inputs.setdefault((task, metric), []).append((value, variance))
            rows.append(
                {
                    "row_type": "pv_specific",
                    "status": status,
                    "model": "EBM_additive",
                    "pv": pv,
                    "task": task,
                    "metric": metric,
                    "estimate": value,
                    "sampling_variance": variance,
                    "training_seconds": classifier_seconds if task == "classification" else regressor_seconds,
                    "peak_rss_delta": after_rss - before_rss,
                    "n_train": len(train_frame),
                    "n_test": len(test_frame),
                    "features": len(features),
                    "sample_weight": "normalized W_FSTUWT",
                }
            )

    for (task, metric), values in pooled_inputs.items():
        estimates, variances = zip(*values)
        pooled = dict(pool_pv_estimates(estimates, variances))
        rows.append(
            {
                "row_type": "pv_pooled",
                "status": status,
                "model": "EBM_additive",
                "pv": "PV1MATH-PV10MATH",
                "task": task,
                "metric": metric,
                **pooled,
                "ci_lower": pooled["estimate"] - 1.96 * pooled["standard_error"],
                "ci_upper": pooled["estimate"] + 1.96 * pooled["standard_error"],
                "n_train": len(train_frame),
                "n_test": len(test_frame),
                "features": len(features),
                "sample_weight": "normalized W_FSTUWT",
            }
        )

    result = pd.DataFrame(rows)
    ensure_v5_output_path(output_path)
    result.to_csv(output_path, index=False)
    manifest = {
        "status": status,
        "interpret_version": interpret.__version__,
        "python": platform.python_version(),
        "config": model_config,
        "n_train": int(len(train_frame)),
        "n_test": int(len(test_frame)),
        "features": features,
        "weighting": "normalized W_FSTUWT on both EBM tasks",
        "outcome": "per-PV regression and I(PVvMATH < 420.07) model-evaluation target",
        "inference_boundary": "Fay-BRR fixed-model evaluation uncertainty; not full training uncertainty",
        "elapsed_seconds": round(perf_counter() - start, 2),
        "output": str(output_path),
    }
    manifest_path = output_path.with_name("v5_controlled_ebm_manifest.json")
    ensure_v5_output_path(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
