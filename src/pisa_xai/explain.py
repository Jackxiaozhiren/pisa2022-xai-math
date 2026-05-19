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


def compute_ale(
    model,
    x,
    features: List[str],
    n_bins: int = 20,
    max_rows: int = 10000,
) -> Optional[dict]:
    """Compute first-order ALE (Accumulated Local Effects) for specified features.

    ALE is preferred over Partial Dependence when features are correlated,
    which is common in PISA data (e.g., ICT resources and SES).

    Returns a dict with keys "ale_values" (dict of feature -> ale array),
    "bin_centers" (dict of feature -> bin center array), and "feature_names".
    """
    require_package("numpy", "pip install -r requirements.txt")
    import numpy as np

    preprocessor = model.named_steps.get("preprocess")
    estimator = model.named_steps.get("model")
    if preprocessor is None or estimator is None:
        return None

    if len(x) > max_rows:
        x_sample = x.sample(max_rows, random_state=20260510)
    else:
        x_sample = x

    transformed = preprocessor.transform(x_sample)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    try:
        feature_names = np.array(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = np.array([f"feature_{idx}" for idx in range(transformed.shape[1])])

    transformed_df = np.asarray(transformed)

    ale_values = {}
    bin_centers = {}

    for feature_name in features:
        idx = None
        for i, name in enumerate(feature_names):
            if feature_name in str(name):
                idx = i
                break
        if idx is None:
            continue

        feat_vals = transformed_df[:, idx]
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(feat_vals, percentiles)
        if len(np.unique(bin_edges)) < 3:
            continue
        bin_edges = np.unique(bin_edges)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        ale = np.zeros(len(centers))
        counts = np.zeros(len(centers))

        for k in range(len(centers)):
            lower = bin_edges[k]
            upper = bin_edges[k + 1]
            in_bin = (feat_vals >= lower) & (feat_vals < upper)
            if k == len(centers) - 1:
                in_bin = (feat_vals >= lower) & (feat_vals <= upper)
            counts[k] = in_bin.sum()
            if counts[k] < 2:
                continue

            x_lower = transformed_df[in_bin].copy()
            x_upper = transformed_df[in_bin].copy()
            x_lower[:, idx] = lower
            x_upper[:, idx] = upper

            try:
                y_lower = estimator.predict(x_lower)
                y_upper = estimator.predict(x_upper)
            except Exception:
                try:
                    y_lower = estimator.predict_proba(x_lower)[:, 1]
                    y_upper = estimator.predict_proba(x_upper)[:, 1]
                except Exception:
                    continue

            ale[k] = np.mean(y_upper - y_lower)

        ale = np.cumsum(ale)
        ale = ale - np.average(ale, weights=np.maximum(counts, 1e-10))
        ale_values[feature_name] = ale
        bin_centers[feature_name] = centers

    return {
        "ale_values": ale_values,
        "bin_centers": bin_centers,
        "feature_names": list(feature_names),
    }


def compute_second_order_ale(
    model,
    x,
    feature_pairs: List[Tuple[str, str]],
    n_bins: int = 10,
    max_rows: int = 5000,
) -> Optional[dict]:
    """Compute second-order ALE for specified feature pairs.

    Returns a dict mapping (feat_a, feat_b) -> {"ale_values": 2d array,
    "bin_centers": (array, array)}.
    """
    require_package("numpy", "pip install -r requirements.txt")
    import numpy as np

    preprocessor = model.named_steps.get("preprocess")
    estimator = model.named_steps.get("model")
    if preprocessor is None or estimator is None:
        return None

    if len(x) > max_rows:
        x_sample = x.sample(max_rows, random_state=20260510)
    else:
        x_sample = x

    transformed = preprocessor.transform(x_sample)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    try:
        feature_names = np.array(preprocessor.get_feature_names_out())
    except Exception:
        return None

    transformed_df = np.asarray(transformed)
    results = {}

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

        vals_a = transformed_df[:, idx_a]
        vals_b = transformed_df[:, idx_b]
        percentiles = np.linspace(0, 100, n_bins + 1)
        edges_a = np.unique(np.percentile(vals_a, percentiles))
        edges_b = np.unique(np.percentile(vals_b, percentiles))
        if len(edges_a) < 2 or len(edges_b) < 2:
            continue

        centers_a = (edges_a[:-1] + edges_a[1:]) / 2
        centers_b = (edges_b[:-1] + edges_b[1:]) / 2
        ale_2d = np.zeros((len(centers_a), len(centers_b)))

        for i in range(len(centers_a)):
            for j in range(len(centers_b)):
                low_a, high_a = edges_a[i], edges_a[i + 1]
                low_b, high_b = edges_b[j], edges_b[j + 1]
                in_bin = (vals_a >= low_a) & (vals_a < high_a) & (vals_b >= low_b) & (vals_b < high_b)
                if in_bin.sum() < 4:
                    continue

                x_ll = transformed_df[in_bin].copy()
                x_lh = transformed_df[in_bin].copy()
                x_hl = transformed_df[in_bin].copy()
                x_hh = transformed_df[in_bin].copy()
                x_ll[:, idx_a] = low_a
                x_ll[:, idx_b] = low_b
                x_lh[:, idx_a] = low_a
                x_lh[:, idx_b] = high_b
                x_hl[:, idx_a] = high_a
                x_hl[:, idx_b] = low_b
                x_hh[:, idx_a] = high_a
                x_hh[:, idx_b] = high_b

                try:
                    pred_ll = np.mean(estimator.predict(x_ll))
                    pred_lh = np.mean(estimator.predict(x_lh))
                    pred_hl = np.mean(estimator.predict(x_hl))
                    pred_hh = np.mean(estimator.predict(x_hh))
                except Exception:
                    try:
                        pred_ll = np.mean(estimator.predict_proba(x_ll)[:, 1])
                        pred_lh = np.mean(estimator.predict_proba(x_lh)[:, 1])
                        pred_hl = np.mean(estimator.predict_proba(x_hl)[:, 1])
                        pred_hh = np.mean(estimator.predict_proba(x_hh)[:, 1])
                    except Exception:
                        continue

                delta_a = pred_hl - pred_ll + pred_hh - pred_lh
                ale_2d[i, j] = delta_a / 2.0

        results[(feat_a, feat_b)] = {
            "ale_values": ale_2d,
            "bin_centers_a": centers_a,
            "bin_centers_b": centers_b,
        }

    return results


def save_ale_plots(
    ale_result: dict,
    output_dir: str | Path,
    top_n: int = 12,
) -> List[Path]:
    """Generate ALE first-order plots from compute_ale() results."""
    require_package("matplotlib", "pip install -r requirements.txt")
    import matplotlib.pyplot as plt
    import pandas as pd

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    ale_values = ale_result["ale_values"]
    bin_centers = ale_result["bin_centers"]

    ale_summary = []
    for feat in ale_values:
        ale_arr = ale_values[feat]
        ale_summary.append({
            "feature": feat,
            "ale_range": float(ale_arr.max()) - float(ale_arr.min()),
            "ale_abs_mean": float(np.abs(ale_arr).mean()),
        })

    ranked = sorted(ale_summary, key=lambda r: r["ale_abs_mean"], reverse=True)[:top_n]

    for rank_info in ranked:
        feat = rank_info["feature"]
        ale_arr = ale_values[feat]
        centers = bin_centers[feat]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(centers, ale_arr, linewidth=2, color="#1f77b4")
        ax.fill_between(centers, 0, ale_arr, alpha=0.15, color="#1f77b4")
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel(feat, fontsize=11)
        ax.set_ylabel("ALE (centered)", fontsize=11)
        ax.set_title(f"ALE: {feat}", fontsize=12)
        fig.tight_layout()
        fname = out_dir / f"ale_{feat}.png"
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(fname)

    ale_df = pd.DataFrame(ale_summary)
    table_path = out_dir / "ale_importance.csv"
    ale_df.sort_values("ale_abs_mean", ascending=False).to_csv(table_path, index=False)

    return saved


def save_shap_vs_ale_comparison(
    shap_feature_importance_df,
    ale_result: dict,
    output_path: str | Path,
) -> Optional[Path]:
    """Create a side-by-side SHAP vs ALE importance comparison chart."""
    require_package("matplotlib", "pip install -r requirements.txt")
    require_package("pandas", "pip install -r requirements.txt")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    ale_values = ale_result["ale_values"]
    ale_importance = pd.DataFrame([
        {"feature": feat, "ale_abs_mean": float(np.abs(vals).mean())}
        for feat, vals in ale_values.items()
    ]).sort_values("ale_abs_mean", ascending=False)

    shap_importance = shap_feature_importance_df.copy()
    if "importance_mean" in shap_importance.columns:
        shap_importance = shap_importance.rename(
            columns={"importance_mean": "shap_importance"}
        )
    shap_importance = shap_importance.sort_values(
        by=[c for c in ["shap_importance", "importance_mean"] if c in shap_importance.columns],
        ascending=False,
    )

    if "feature" not in shap_importance.columns:
        return None

    merged = pd.merge(
        shap_importance[["feature", "shap_importance"] if "shap_importance" in shap_importance.columns else ["feature", "importance_mean"]],
        ale_importance,
        on="feature",
        how="outer",
    ).fillna(0)

    merged["shap_rank"] = merged["shap_importance"].rank(ascending=False) if "shap_importance" in merged.columns else merged["importance_mean"].rank(ascending=False)
    merged["ale_rank"] = merged["ale_abs_mean"].rank(ascending=False)
    merged = merged.head(15)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(5, 0.35 * len(merged))))
    merged_sorted_shap = merged.sort_values("shap_rank", ascending=True)
    ax1.barh(range(len(merged_sorted_shap)), merged_sorted_shap["shap_rank"].values, color="#1f77b4", alpha=0.7)
    ax1.set_yticks(range(len(merged_sorted_shap)))
    ax1.set_yticklabels(merged_sorted_shap["feature"].values)
    ax1.set_xlabel("SHAP Rank (lower = more important)")
    ax1.set_title("SHAP Importance Ranking")
    ax1.invert_yaxis()

    merged_sorted_ale = merged.sort_values("ale_rank", ascending=True)
    ax2.barh(range(len(merged_sorted_ale)), merged_sorted_ale["ale_rank"].values, color="#ff7f0e", alpha=0.7)
    ax2.set_yticks(range(len(merged_sorted_ale)))
    ax2.set_yticklabels(merged_sorted_ale["feature"].values)
    ax2.set_xlabel("ALE Rank (lower = more important)")
    ax2.set_title("ALE Importance Ranking")
    ax2.invert_yaxis()

    fig.suptitle("SHAP vs ALE Feature Importance Rankings", fontsize=13, y=1.01)
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


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
