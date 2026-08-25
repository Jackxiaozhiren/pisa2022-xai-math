from __future__ import annotations

from math import sqrt
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split


class MetricNotComputable(ValueError):
    """Raised when a survey metric has no valid definition for an input subset."""


def controlled_ebm_config(n_jobs: int) -> Mapping[str, object]:
    if n_jobs < 1:
        raise ValueError("n_jobs must be positive")
    return {
        "interactions": 0,
        "max_bins": 256,
        "outer_bags": 1,
        "inner_bags": 0,
        "learning_rate": 0.01,
        "max_rounds": 1000,
        "early_stopping_rounds": 50,
        "min_samples_leaf": 10,
        "random_state": 20260510,
        "n_jobs": n_jobs,
    }


def ensure_v5_output_path(path: str | Path) -> Path:
    output_path = Path(path)
    if not output_path.name.startswith("v5_"):
        raise ValueError("v5 candidate outputs must use a v5_ filename")
    return output_path


def validate_weights(values: Sequence[float] | pd.Series | np.ndarray) -> np.ndarray:
    weights = np.asarray(values, dtype=float)
    if weights.size == 0:
        raise ValueError("weights are empty")
    if not np.isfinite(weights).all():
        raise ValueError("weights contain non-finite values")
    if (weights < 0).any():
        raise ValueError("weights contain negative values")
    if weights.sum() <= 0:
        raise ValueError("weights must have a positive total")
    return weights


def validate_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    keys: Sequence[str],
    matching_columns: Sequence[str],
) -> pd.DataFrame:
    missing_left = [column for column in [*keys, *matching_columns] if column not in left.columns]
    missing_right = [column for column in [*keys, *matching_columns] if column not in right.columns]
    if missing_left or missing_right:
        raise ValueError(f"join columns missing: left={missing_left}, right={missing_right}")
    if left.duplicated(list(keys)).any() or right.duplicated(list(keys)).any():
        raise ValueError("join keys must be unique on both sides")

    comparison = left[[*keys, *matching_columns]].merge(
        right[[*keys, *matching_columns]],
        on=list(keys),
        how="left",
        validate="one_to_one",
        suffixes=("_left", "_right"),
    )
    if len(comparison) != len(left) or comparison[[f"{column}_right" for column in matching_columns]].isna().any().any():
        raise ValueError("join changed rows or left unmatched keys")
    for column in matching_columns:
        left_values = comparison[f"{column}_left"].to_numpy()
        right_values = comparison[f"{column}_right"].to_numpy()
        if not np.allclose(left_values, right_values, rtol=0, atol=1e-12, equal_nan=True):
            raise ValueError(f"join mismatch for {column}")

    payload = right.drop(columns=list(matching_columns), errors="ignore")
    joined = left.merge(payload, on=list(keys), how="left", validate="one_to_one")
    if len(joined) != len(left):
        raise ValueError("join changed row count")
    return joined


def fixed_legacy_split_indices(
    index: pd.Index,
    legacy_labels: pd.Series,
    test_size: float = 0.2,
    random_state: int = 20260510,
) -> tuple[list[int], list[int]]:
    train_index, test_index = train_test_split(
        list(index),
        test_size=test_size,
        random_state=random_state,
        stratify=legacy_labels.loc[index],
    )
    return list(train_index), list(test_index)


def institutional_cold_start_split_indices(
    frame: pd.DataFrame,
    country_column: str = "CNT",
    school_column: str = "CNTSCHID",
    test_fraction: float = 0.2,
    random_state: int = 20260510,
) -> tuple[list[int], list[int], pd.DataFrame]:
    """Split whole schools within each country for unseen-institution validation.

    The split is deterministic across Python processes: school IDs are sorted and
    each country receives a separate NumPy generator seeded by its sorted-country
    rank. The returned audit frame records train/test school counts by country.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    missing = [c for c in (country_column, school_column) if c not in frame.columns]
    if missing:
        raise ValueError(f"institution split columns missing: {missing}")
    keys = frame[[country_column, school_column]].copy()
    if keys.isna().any().any():
        raise ValueError("institution split identifiers must be non-missing")
    keys[country_column] = keys[country_column].astype(str)
    keys[school_column] = pd.to_numeric(keys[school_column], errors="raise")
    pairs = keys.drop_duplicates().sort_values([country_column, school_column])
    test_pairs: set[tuple[str, int | float]] = set()
    audit_rows: list[dict[str, object]] = []
    for country_rank, country in enumerate(pairs[country_column].unique()):
        schools = pairs.loc[pairs[country_column] == country, school_column].tolist()
        if len(schools) < 2:
            raise ValueError(f"country {country!r} has fewer than two schools")
        rng = np.random.default_rng(random_state + country_rank * 1009)
        order = rng.permutation(len(schools))
        n_test = max(1, int(round(len(schools) * test_fraction)))
        n_test = min(n_test, len(schools) - 1)
        selected = {schools[int(i)] for i in order[:n_test]}
        test_pairs.update((str(country), school) for school in selected)
        audit_rows.append(
            {
                "country": str(country),
                "schools_total": len(schools),
                "schools_train": len(schools) - n_test,
                "schools_test": n_test,
            }
        )
    pair_series = list(zip(keys[country_column], keys[school_column]))
    test_mask = np.array([pair in test_pairs for pair in pair_series], dtype=bool)
    train_index = frame.index[~test_mask].tolist()
    test_index = frame.index[test_mask].tolist()
    audit = pd.DataFrame(audit_rows).sort_values("country").reset_index(drop=True)
    return train_index, test_index, audit


def require_output_columns(frame: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"output schema missing columns: {missing}")


def fay_brr_variance(full_estimate: float, replicate_estimates: Iterable[float]) -> float:
    replicates = np.asarray(list(replicate_estimates), dtype=float)
    if replicates.size != 80:
        raise ValueError("PISA Fay-BRR requires exactly 80 replicate estimates")
    if not np.isfinite(replicates).all() or not np.isfinite(full_estimate):
        raise ValueError("replicate estimates must be finite")
    return float(0.05 * np.square(replicates - full_estimate).sum())


def pool_pv_estimates(
    estimates: Sequence[float], sampling_variances: Sequence[float]
) -> Mapping[str, float]:
    estimate_array = np.asarray(estimates, dtype=float)
    variance_array = np.asarray(sampling_variances, dtype=float)
    if estimate_array.size != 10 or variance_array.size != 10:
        raise ValueError("PISA 2022 pooling requires exactly ten PV estimates and variances")
    if not np.isfinite(estimate_array).all() or not np.isfinite(variance_array).all():
        raise ValueError("PV estimates and variances must be finite")
    estimate = float(estimate_array.mean())
    sampling_variance = float(variance_array.mean())
    imputation_variance = float(estimate_array.var(ddof=1))
    total_variance = sampling_variance + (1 + 1 / 10) * imputation_variance
    return {
        "estimate": estimate,
        "sampling_variance": sampling_variance,
        "imputation_variance": imputation_variance,
        "total_variance": total_variance,
        "standard_error": sqrt(total_variance),
    }


def weighted_ece(
    y_true: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    sample_weight: Sequence[float] | np.ndarray,
    bins: int = 10,
) -> float:
    if bins < 1:
        raise ValueError("bins must be positive")
    outcomes = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    weights = validate_weights(sample_weight)
    if not (len(outcomes) == len(scores) == len(weights)):
        raise ValueError("outcomes, probabilities, and weights must have the same length")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("probabilities must be finite values in [0, 1]")
    bin_index = np.minimum((scores * bins).astype(int), bins - 1)
    total_weight = weights.sum()
    ece = 0.0
    for index in range(bins):
        mask = bin_index == index
        if not mask.any():
            continue
        bin_weight = weights[mask].sum()
        observed = np.average(outcomes[mask], weights=weights[mask])
        predicted = np.average(scores[mask], weights=weights[mask])
        ece += (bin_weight / total_weight) * abs(predicted - observed)
    return float(ece)


def _calibration_slope(
    y_true: np.ndarray, probabilities: np.ndarray, sample_weight: np.ndarray
) -> float:
    if np.unique(y_true).size != 2:
        raise MetricNotComputable("calibration slope requires both classes")
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped))
    if np.isclose(np.ptp(logits), 0):
        raise MetricNotComputable("calibration slope requires varying probabilities")
    calibrator = LogisticRegression(C=1_000_000.0, max_iter=1000, solver="lbfgs")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        calibrator.fit(logits.reshape(-1, 1), y_true, sample_weight=sample_weight)
    slope = float(calibrator.coef_[0, 0])
    if abs(slope) > 100:
        raise MetricNotComputable("calibration slope indicates numerical separation")
    if not np.isfinite(slope):
        raise MetricNotComputable("calibration slope is non-finite")
    return slope


def weighted_binary_metrics(
    y_true: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    sample_weight: Sequence[float] | np.ndarray,
) -> Mapping[str, float]:
    outcomes = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    weights = validate_weights(sample_weight)
    if not (len(outcomes) == len(scores) == len(weights)):
        raise ValueError("outcomes, probabilities, and weights must have the same length")
    if np.unique(outcomes).size != 2:
        raise MetricNotComputable("binary metrics require both classes")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("probabilities must be finite values in [0, 1]")
    return {
        "auc": float(roc_auc_score(outcomes, scores, sample_weight=weights)),
        "brier": float(brier_score_loss(outcomes, scores, sample_weight=weights)),
        "ece": weighted_ece(outcomes, scores, weights),
        "calibration_slope": _calibration_slope(outcomes, scores, weights),
    }
