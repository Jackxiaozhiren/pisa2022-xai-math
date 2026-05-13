from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from .io import require_package
from .pisa import LOW_PERFORMER_MATH_LEVEL_2_THRESHOLD, math_pv_columns


def flatten_feature_config(feature_config: Dict[str, Iterable[str]]) -> List[str]:
    seen = set()
    flattened: List[str] = []
    for group_vars in feature_config.values():
        for name in group_vars:
            if name not in seen:
                flattened.append(name)
                seen.add(name)
    return flattened


def feature_group_lookup(feature_config: Dict[str, Iterable[str]]) -> Dict[str, str]:
    """Return the first configured conceptual group for each feature."""

    lookup: Dict[str, str] = {}
    for group, group_vars in feature_config.items():
        for name in group_vars:
            lookup.setdefault(name, group)
    return lookup


def available_columns(df, requested: Iterable[str]) -> Tuple[List[str], List[str]]:
    columns = set(df.columns)
    available = [name for name in requested if name in columns]
    missing = [name for name in requested if name not in columns]
    return available, missing


def add_math_targets(
    df,
    pv_count: int = 10,
    low_threshold: float = LOW_PERFORMER_MATH_LEVEL_2_THRESHOLD,
):
    require_package("pandas", "pip install -r requirements.txt")
    pv_cols = [col for col in math_pv_columns(pv_count) if col in df.columns]
    if not pv_cols:
        raise ValueError("No mathematics plausible-value columns were found.")
    result = df.copy()
    result["MATH_PV_MEAN"] = result[pv_cols].mean(axis=1)
    result["LOW_PERFORMER_MATH"] = (result["MATH_PV_MEAN"] < low_threshold).astype(int)
    return result


def select_model_frame(df, feature_names: Iterable[str], extra_columns: Iterable[str]):
    requested = list(feature_names) + list(extra_columns)
    available, missing = available_columns(df, requested)
    return df[available].copy(), missing
