from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

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


def save_shap_dependence_plots(
    model,
    x,
    output_dir: str | Path,
    feature_pairs: List[Tuple[str, str]],
    max_rows: int = 3000,
) -> List[Path]:
    require_package("numpy", "pip install -r requirements.txt")
    require_package("matplotlib", "pip install -r requirements.txt")
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        import shap
    except Exception:
        return []

    if len(x) > max_rows:
        x_sample = x.sample(max_rows, random_state=20260510)
    else:
        x_sample = x

    preprocessor = model.named_steps.get("preprocess")
    estimator = model.named_steps.get("model")
    if preprocessor is None or estimator is None:
        return []

    transformed = preprocessor.transform(x_sample)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    try:
        feature_names = np.array(preprocessor.get_feature_names_out())
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
        return []

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for feat_a, feat_b in feature_pairs:
        idx_a = None
        idx_b = None
        for i, name in enumerate(feature_names):
            if feat_a in str(name):
                idx_a = i
            if feat_b in str(name):
                idx_b = i
        if idx_a is None or idx_b is None:
            continue

        plt.figure(figsize=(8, 6))
        try:
            shap.plots.scatter(
                shap_values[:, idx_a],
                color=shap_values[:, idx_b],
                show=False,
            )
        except Exception:
            try:
                shap.dependence_plot(
                    idx_a,
                    shap_values.values if hasattr(shap_values, "values") else shap_values,
                    transformed,
                    feature_names=feature_names,
                    interaction_index=idx_b,
                    show=False,
                )
            except Exception:
                plt.close()
                continue

        plt.title(f"SHAP Dependence: {feat_a} vs {feat_b}")
        fname = out_dir / f"shap_dependence_{feat_a}_vs_{feat_b}.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
        saved.append(fname)

    return saved


def compute_shap_interactions(
    model,
    x,
    max_rows: int = 2000,
) -> Optional[object]:
    require_package("numpy", "pip install -r requirements.txt")
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
        shap_interaction = explainer(transformed)
        if getattr(shap_interaction, "values", None) is not None and shap_interaction.values.ndim == 3:
            interaction_values = shap_interaction.values[:, :, 1]
        else:
            interaction_values = shap_interaction.values if hasattr(shap_interaction, "values") else None

        if interaction_values is None:
            return None

        return {
            "interaction_values": interaction_values,
            "feature_names": feature_names,
            "x_sample": x_sample,
            "transformed": transformed,
        }
    except Exception:
        return None
