from __future__ import annotations

from pathlib import Path
from typing import Optional

from .io import require_package


def permutation_importance_table(model, x, y, scoring: str, n_repeats: int = 10):
    require_package("pandas", "pip install -r requirements.txt")
    require_package("sklearn", "pip install -r requirements.txt")
    import pandas as pd
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model,
        x,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=20260510,
        n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": x.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def save_shap_summary(model, x, output_path: str | Path, max_rows: int = 5000) -> Optional[Path]:
    """Save a SHAP beeswarm plot when shap is available.

    This expects a fitted sklearn Pipeline. If SHAP cannot explain the fitted model,
    callers should use permutation importance as a fallback.
    """

    require_package("numpy", "pip install -r requirements.txt")
    require_package("matplotlib", "pip install -r requirements.txt")
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        import shap
    except Exception:
        return None

    if len(x) > max_rows:
        x_sample = x.sample(max_rows, random_state=20260510)
    else:
        x_sample = x

    preprocessor = model.named_steps.get("preprocess")
    estimator = model.named_steps.get("model")
    if preprocessor is None or estimator is None:
        return None

    transformed = preprocessor.transform(x_sample)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = np.array([f"feature_{idx}" for idx in range(transformed.shape[1])])

    try:
        explainer = shap.Explainer(estimator, transformed)
        try:
            shap_values = explainer(transformed, check_additivity=False)
        except TypeError:
            shap_values = explainer(transformed)
        if getattr(shap_values, "values", None) is not None and shap_values.values.ndim == 3:
            shap_values = shap_values[:, :, 1]
    except Exception:
        return None

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(shap_values, transformed, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    return output
