from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import mean
from typing import Iterable, List, Sequence


LOW_PERFORMER_MATH_LEVEL_2_THRESHOLD = 420.07


@dataclass(frozen=True)
class PlausibleValueEstimate:
    estimate: float
    sampling_variance: float
    imputation_variance: float
    total_variance: float
    standard_error: float


def math_pv_columns(count: int = 10, prefix: str = "PV", suffix: str = "MATH") -> List[str]:
    return [f"{prefix}{idx}{suffix}" for idx in range(1, count + 1)]


def replicate_weight_columns(prefix: str = "W_FSTURWT", count: int = 80) -> List[str]:
    return [f"{prefix}{idx}" for idx in range(1, count + 1)]


def brr_standard_error(full_sample_estimate: float, replicate_estimates: Iterable[float]) -> float:
    """Return the PISA BRR standard error.

    PISA 2022 uses 80 replicate weights and a Fay factor that yields the variance
    formula 0.05 * sum((replicate - full_sample)^2).
    """

    variance = 0.05 * sum((estimate - full_sample_estimate) ** 2 for estimate in replicate_estimates)
    return sqrt(variance)


def combine_plausible_value_estimates(
    estimates: Sequence[float],
    sampling_variances: Sequence[float],
) -> PlausibleValueEstimate:
    """Pool estimates across plausible values using standard multiple-imputation logic."""

    if not estimates:
        raise ValueError("At least one plausible-value estimate is required.")
    if len(estimates) != len(sampling_variances):
        raise ValueError("Estimates and sampling variances must have the same length.")

    m = len(estimates)
    pooled_estimate = mean(estimates)
    sampling_variance = mean(sampling_variances)
    if m == 1:
        imputation_variance = 0.0
    else:
        imputation_variance = sum((estimate - pooled_estimate) ** 2 for estimate in estimates) / (m - 1)
    total_variance = sampling_variance + (1 + 1 / m) * imputation_variance
    return PlausibleValueEstimate(
        estimate=pooled_estimate,
        sampling_variance=sampling_variance,
        imputation_variance=imputation_variance,
        total_variance=total_variance,
        standard_error=sqrt(total_variance),
    )


def low_performer_flag(score: float, threshold: float = LOW_PERFORMER_MATH_LEVEL_2_THRESHOLD) -> int:
    return int(score < threshold)
