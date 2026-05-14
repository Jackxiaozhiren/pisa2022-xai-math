from __future__ import annotations

from typing import Dict, Optional

from .io import require_package


def regression_metrics(y_true, y_pred, sample_weight: Optional[object] = None) -> Dict[str, float]:
    require_package("sklearn", "pip install -r requirements.txt")
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mse = mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
    rmse = mse**0.5
    return {
        "rmse": float(rmse),
        "mae": float(mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)),
        "r2": float(r2_score(y_true, y_pred, sample_weight=sample_weight)),
    }


def classification_metrics(
    y_true,
    y_score,
    threshold: float = 0.5,
    sample_weight: Optional[object] = None,
) -> Dict[str, float]:
    require_package("numpy", "pip install -r requirements.txt")
    require_package("sklearn", "pip install -r requirements.txt")
    import numpy as np
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_score, sample_weight=sample_weight)),
        "average_precision": float(
            average_precision_score(y_true, y_score, sample_weight=sample_weight)
        ),
        "f1": float(f1_score(y_true, y_pred, sample_weight=sample_weight)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0, sample_weight=sample_weight)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0, sample_weight=sample_weight)
        ),
        "brier": float(brier_score_loss(y_true, y_score, sample_weight=sample_weight)),
    }


def calibration_summary(
    y_true,
    y_score,
    sample_weight: Optional[object] = None,
) -> Dict[str, float]:
    require_package("numpy", "pip install -r requirements.txt")
    require_package("sklearn", "pip install -r requirements.txt")
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss

    y_score = np.asarray(y_score, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

    eps = 1e-6
    clipped = np.clip(y_score, eps, 1 - eps)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)

    intercept = float("nan")
    slope = float("nan")
    if len(np.unique(y_true)) == 2:
        calibrator = LogisticRegression(C=1_000_000.0, max_iter=1000, solver="lbfgs")
        calibrator.fit(logits, y_true, sample_weight=weights)
        intercept = float(calibrator.intercept_[0])
        slope = float(calibrator.coef_[0][0])

    if weights is None:
        mean_predicted = float(np.mean(y_score))
        observed_rate = float(np.mean(y_true))
    else:
        mean_predicted = float(np.average(y_score, weights=weights))
        observed_rate = float(np.average(y_true, weights=weights))

    return {
        "brier": float(brier_score_loss(y_true, y_score, sample_weight=weights)),
        "mean_predicted_probability": mean_predicted,
        "observed_low_performer_rate": observed_rate,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def calibration_bins(
    y_true,
    y_score,
    sample_weight: Optional[object] = None,
    n_bins: int = 10,
):
    require_package("numpy", "pip install -r requirements.txt")
    require_package("pandas", "pip install -r requirements.txt")
    import numpy as np
    import pandas as pd

    y_score = np.asarray(y_score, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    weights = np.ones_like(y_score, dtype=float) if sample_weight is None else np.asarray(
        sample_weight,
        dtype=float,
    )
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_score, edges[1:-1], right=True)

    rows = []
    expected_calibration_error = 0.0
    total_weight = float(weights.sum())
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not mask.any():
            rows.append(
                {
                    "bin": bin_id + 1,
                    "score_min": float(edges[bin_id]),
                    "score_max": float(edges[bin_id + 1]),
                    "n": 0,
                    "weight_sum": 0.0,
                    "mean_predicted_probability": float("nan"),
                    "observed_low_performer_rate": float("nan"),
                    "absolute_calibration_gap": float("nan"),
                }
            )
            continue
        bin_weight = weights[mask]
        weight_sum = float(bin_weight.sum())
        mean_predicted = float(np.average(y_score[mask], weights=bin_weight))
        observed = float(np.average(y_true[mask], weights=bin_weight))
        gap = abs(mean_predicted - observed)
        expected_calibration_error += (weight_sum / total_weight) * gap
        rows.append(
            {
                "bin": bin_id + 1,
                "score_min": float(edges[bin_id]),
                "score_max": float(edges[bin_id + 1]),
                "n": int(mask.sum()),
                "weight_sum": weight_sum,
                "mean_predicted_probability": mean_predicted,
                "observed_low_performer_rate": observed,
                "absolute_calibration_gap": float(gap),
            }
        )

    table = pd.DataFrame(rows)
    table.attrs["expected_calibration_error"] = float(expected_calibration_error)
    return table


def threshold_sensitivity(y_true, y_score, sample_weight: Optional[object] = None):
    require_package("numpy", "pip install -r requirements.txt")
    require_package("pandas", "pip install -r requirements.txt")
    require_package("sklearn", "pip install -r requirements.txt")
    import numpy as np
    import pandas as pd
    from sklearn.metrics import precision_recall_curve, roc_curve

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score, sample_weight=sample_weight)
    finite_roc = np.isfinite(roc_thresholds)
    youden_values = (tpr - fpr)[finite_roc]
    youden_threshold = float(roc_thresholds[finite_roc][int(np.argmax(youden_values))])

    precision, recall, pr_thresholds = precision_recall_curve(
        y_true, y_score, sample_weight=sample_weight
    )
    if len(pr_thresholds):
        f1 = (2 * precision[:-1] * recall[:-1]) / np.maximum(
            precision[:-1] + recall[:-1],
            np.finfo(float).eps,
        )
        f1_threshold = float(pr_thresholds[int(np.argmax(f1))])
    else:
        f1_threshold = 0.5

    rows = []
    for label, threshold in [
        ("default_0.50", 0.5),
        ("youden_j", youden_threshold),
        ("max_f1", f1_threshold),
    ]:
        rows.append(
            {
                "threshold_rule": label,
                "threshold": threshold,
                **classification_metrics(y_true, y_score, threshold, sample_weight=sample_weight),
            }
        )
    return pd.DataFrame(rows)


def fairness_metrics(
    y_true: "np.ndarray",
    y_pred: "np.ndarray",
    group_values: "np.ndarray",
    groups: Optional[list] = None,
) -> "pd.DataFrame":
    """Compute group fairness metrics.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted binary labels (thresholded).
    group_values : array-like
        Group membership values (e.g., gender, immigrant status).
    groups : list, optional
        Subset of group values to evaluate. Uses all unique values if None.

    Returns
    -------
    pd.DataFrame with columns: group, n, prevalence, TPR, FPR,
    selection_rate, dp_diff, eo_diff.
    """
    import numpy as np
    import pandas as pd

    require_package("numpy", "pip install -r requirements.txt")

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    group_values = np.asarray(group_values)

    if groups is None:
        groups = sorted(set(group_values))

    rows = []
    baseline_tpr = None
    baseline_fpr = None
    baseline_sel = None

    for i, g in enumerate(groups):
        mask = group_values == g
        n = mask.sum()
        if n == 0:
            continue

        yt = y_true[mask]
        yp = y_pred[mask]

        tp = ((yt == 1) & (yp == 1)).sum()
        tn = ((yt == 0) & (yp == 0)).sum()
        fp = ((yt == 0) & (yp == 1)).sum()
        fn = ((yt == 1) & (yp == 0)).sum()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
        sel_rate = yp.mean()

        row = {
            "group": str(g),
            "n": int(n),
            "prevalence": float(yt.mean()),
            "TPR": float(tpr),
            "FPR": float(fpr),
            "selection_rate": float(sel_rate),
        }
        rows.append(row)

        if i == 0:
            baseline_tpr = tpr
            baseline_fpr = fpr
            baseline_sel = sel_rate

    # Compute differences from baseline (first group)
    for i, row in enumerate(rows):
        if i == 0:
            row["dp_diff"] = 0.0
            row["eo_diff"] = 0.0
        else:
            row["dp_diff"] = float(row["selection_rate"] - baseline_sel)
            row["eo_diff"] = float(
                max(abs(row["TPR"] - baseline_tpr), abs(row["FPR"] - baseline_fpr))
            )

    return pd.DataFrame(rows)
