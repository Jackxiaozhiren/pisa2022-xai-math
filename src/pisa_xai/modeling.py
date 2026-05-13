from __future__ import annotations

from typing import Dict, Iterable, Tuple

from .io import require_package


def split_feature_types(df) -> Tuple[list, list]:
    require_package("pandas", "pip install -r requirements.txt")
    numeric = []
    categorical = []
    for column in df.columns:
        if str(df[column].dtype).startswith(("float", "int", "bool")):
            numeric.append(column)
        else:
            categorical.append(column)
    return numeric, categorical


def make_preprocessor(df):
    require_package("sklearn", "pip install -r requirements.txt")
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric, categorical = split_feature_types(df)
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=25)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical),
        ],
        remainder="drop",
    )


def regression_models(df, enabled_optional_models: Iterable[str] = ()) -> Dict[str, object]:
    require_package("sklearn", "pip install -r requirements.txt")
    from sklearn.base import clone
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.pipeline import Pipeline

    preprocessor = make_preprocessor(df)
    models = {
        "ridge": Ridge(alpha=1.0, random_state=20260510),
        "elastic_net": ElasticNet(
            alpha=0.01,
            l1_ratio=0.2,
            max_iter=3000,
            random_state=20260510,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=120,
            min_samples_leaf=75,
            max_features="sqrt",
            max_samples=0.7,
            n_jobs=-1,
            random_state=20260510,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.06,
            l2_regularization=0.05,
            random_state=20260510,
        ),
    }
    optional = optional_regression_models(enabled_optional_models)
    models.update(optional)
    return {name: Pipeline([("preprocess", clone(preprocessor)), ("model", model)]) for name, model in models.items()}


def classification_models(df, enabled_optional_models: Iterable[str] = ()) -> Dict[str, object]:
    require_package("sklearn", "pip install -r requirements.txt")
    from sklearn.base import clone
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    preprocessor = make_preprocessor(df)
    models = {
        "logistic_l2": LogisticRegression(
            C=1.0,
            max_iter=3000,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=120,
            min_samples_leaf=75,
            max_features="sqrt",
            max_samples=0.7,
            n_jobs=-1,
            random_state=20260510,
            class_weight="balanced_subsample",
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.06,
            l2_regularization=0.05,
            random_state=20260510,
        ),
    }
    optional = optional_classification_models(enabled_optional_models)
    models.update(optional)
    return {name: Pipeline([("preprocess", clone(preprocessor)), ("model", model)]) for name, model in models.items()}


def optional_regression_models(enabled_names: Iterable[str] = ()) -> Dict[str, object]:
    enabled = set(enabled_names)
    models: Dict[str, object] = {}
    if "xgboost" in enabled:
        try:
            from xgboost import XGBRegressor

            models["xgboost"] = XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=20260510,
                n_jobs=-1,
            )
        except Exception:
            pass
    if "lightgbm" in enabled:
        try:
            from lightgbm import LGBMRegressor

            models["lightgbm"] = LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=80,
                random_state=20260510,
                n_jobs=-1,
                verbose=-1,
            )
        except Exception:
            pass
    if "catboost" in enabled:
        try:
            from catboost import CatBoostRegressor

            models["catboost"] = CatBoostRegressor(
                iterations=300,
                learning_rate=0.05,
                depth=6,
                loss_function="RMSE",
                random_seed=20260510,
                verbose=False,
            )
        except Exception:
            pass
    return models


def optional_classification_models(enabled_names: Iterable[str] = ()) -> Dict[str, object]:
    enabled = set(enabled_names)
    models: Dict[str, object] = {}
    if "xgboost" in enabled:
        try:
            from xgboost import XGBClassifier

            models["xgboost"] = XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=20260510,
                n_jobs=-1,
            )
        except Exception:
            pass
    if "lightgbm" in enabled:
        try:
            from lightgbm import LGBMClassifier

            models["lightgbm"] = LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=80,
                random_state=20260510,
                n_jobs=-1,
                verbose=-1,
            )
        except Exception:
            pass
    if "catboost" in enabled:
        try:
            from catboost import CatBoostClassifier

            models["catboost"] = CatBoostClassifier(
                iterations=300,
                learning_rate=0.05,
                depth=6,
                loss_function="Logloss",
                random_seed=20260510,
                verbose=False,
            )
        except Exception:
            pass
    return models
