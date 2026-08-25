from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pisa_xai.v5_survey import (
    MetricNotComputable,
    controlled_ebm_config,
    ensure_v5_output_path,
    fay_brr_variance,
    fixed_legacy_split_indices,
    institutional_cold_start_split_indices,
    pool_pv_estimates,
    require_output_columns,
    validate_join,
    validate_weights,
    weighted_binary_metrics,
    weighted_ece,
)


def test_fay_brr_variance_uses_official_80_replicate_factor() -> None:
    full = 2.0
    replicates = [1.0, 3.0] + [2.0] * 78
    assert fay_brr_variance(full, replicates) == pytest.approx(0.1)


def test_pool_pv_estimates_combines_sampling_and_imputation_variance() -> None:
    pooled = pool_pv_estimates([1.0] * 5 + [3.0] * 5, [0.25] * 10)
    assert pooled["estimate"] == pytest.approx(2.0)
    assert pooled["sampling_variance"] == pytest.approx(0.25)
    assert pooled["imputation_variance"] == pytest.approx(10 / 9)
    assert pooled["total_variance"] == pytest.approx(0.25 + 1.1 * 10 / 9)


def test_validate_weights_rejects_negative_or_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="negative"):
        validate_weights(pd.Series([1.0, -0.1]))
    with pytest.raises(ValueError, match="non-finite"):
        validate_weights(pd.Series([1.0, math.inf]))


def test_validate_join_requires_unique_unchanged_keys_and_matching_columns() -> None:
    left = pd.DataFrame({"CNT": ["A", "B"], "CNTSTUID": [1, 2], "W_FSTUWT": [1.0, 2.0]})
    right = pd.DataFrame({"CNT": ["A", "B"], "CNTSTUID": [1, 2], "W_FSTUWT": [1.0, 2.0], "SENWT": [5.0, 5.0]})
    joined = validate_join(left, right, ["CNT", "CNTSTUID"], ["W_FSTUWT"])
    assert list(joined.columns) == ["CNT", "CNTSTUID", "W_FSTUWT", "SENWT"]
    duplicate = pd.concat([right, right.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        validate_join(left, duplicate, ["CNT", "CNTSTUID"], ["W_FSTUWT"])


def test_weighted_ece_matches_two_bin_known_case() -> None:
    result = weighted_ece(
        np.array([0, 1]),
        np.array([0.2, 0.8]),
        np.array([1.0, 1.0]),
        bins=2,
    )
    assert result == pytest.approx(0.2)


def test_binary_metrics_refuse_single_class_subgroup() -> None:
    with pytest.raises(MetricNotComputable, match="both classes"):
        weighted_binary_metrics(
            np.array([1, 1, 1]),
            np.array([0.8, 0.9, 0.7]),
            np.array([1.0, 1.0, 1.0]),
        )


def test_binary_metrics_handles_extreme_probabilities_without_crashing() -> None:
    metrics = weighted_binary_metrics(
        np.array([0, 1, 0, 1]),
        np.array([0.0, 1.0, 1e-15, 1.0 - 1e-15]),
        np.ones(4),
    )
    assert metrics["auc"] == pytest.approx(1.0)
    assert math.isfinite(metrics["calibration_slope"])


def test_v5_output_guard_protects_legacy_files() -> None:
    assert ensure_v5_output_path("reports/tables/v5_pv_specific_metrics.csv").name.startswith("v5_")
    with pytest.raises(ValueError, match="v5_"):
        ensure_v5_output_path("reports/tables/model_metrics.csv")


def test_fixed_legacy_split_is_seed_deterministic() -> None:
    index = pd.Index(range(20))
    labels = pd.Series([0, 1] * 10, index=index)
    first = fixed_legacy_split_indices(index, labels)
    second = fixed_legacy_split_indices(index, labels)
    assert first == second
    assert len(first[1]) == 4


def test_institutional_cold_start_split_has_no_school_overlap_and_is_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "CNT": ["A"] * 12 + ["B"] * 12,
            "CNTSCHID": [1] * 4 + [2] * 4 + [3] * 4 + [10] * 4 + [11] * 4 + [12] * 4,
        }
    )
    train_a, test_a, audit_a = institutional_cold_start_split_indices(frame, test_fraction=0.5)
    train_b, test_b, audit_b = institutional_cold_start_split_indices(frame, test_fraction=0.5)
    assert train_a == train_b and test_a == test_b
    assert audit_a.equals(audit_b)
    train_pairs = set(zip(frame.loc[train_a, "CNT"], frame.loc[train_a, "CNTSCHID"]))
    test_pairs = set(zip(frame.loc[test_a, "CNT"], frame.loc[test_a, "CNTSCHID"]))
    assert train_pairs.isdisjoint(test_pairs)
    assert set(audit_a["schools_train"]) == {1}
    assert set(audit_a["schools_test"]) == {2}


def test_output_schema_requires_all_publication_fields() -> None:
    required = ["pv", "estimand", "metric", "estimate"]
    require_output_columns(pd.DataFrame([{field: 1 for field in required}]), required)
    with pytest.raises(ValueError, match="missing"):
        require_output_columns(pd.DataFrame([{"pv": 1, "estimand": "population"}]), required)


def test_controlled_ebm_config_locks_additive_full_data_comparison() -> None:
    config = controlled_ebm_config(n_jobs=6)
    assert config["interactions"] == 0
    assert config["outer_bags"] == 1
    assert config["random_state"] == 20260510
    assert config["n_jobs"] == 6
