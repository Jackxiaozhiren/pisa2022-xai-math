from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

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


# ── Hyperparameter Tuning with Optuna ────────────────────────────────────────


def _default_optuna_study_kwargs() -> Dict[str, Any]:
    return {"direction": "minimize", "sampler": None, "pruner": None}


def tune_lightgbm_regressor(
    x_train,
    y_train,
    x_val,
    y_val,
    n_trials: int = 50,
    random_state: int = 20260510,
) -> Dict[str, Any]:
    require_package("lightgbm", "pip install lightgbm")
    require_package("numpy", "pip install -r requirements.txt")
    require_package("sklearn", "pip install -r requirements.txt")
    import numpy as np
    from lightgbm import LGBMRegressor
    from sklearn.metrics import mean_squared_error

    try:
        import optuna
    except Exception:
        return {}

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = LGBMRegressor(**params)
        model.fit(x_train, y_train)
        preds = model.predict(x_val)
        return float(np.sqrt(mean_squared_error(y_val, preds)))

    study = optuna.create_study(**_default_optuna_study_kwargs())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_lightgbm_classifier(
    x_train,
    y_train,
    x_val,
    y_val,
    n_trials: int = 50,
    random_state: int = 20260510,
) -> Dict[str, Any]:
    require_package("lightgbm", "pip install lightgbm")
    require_package("numpy", "pip install -r requirements.txt")
    require_package("sklearn", "pip install -r requirements.txt")
    import numpy as np
    from lightgbm import LGBMClassifier
    from sklearn.metrics import log_loss

    try:
        import optuna
    except Exception:
        return {}

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = LGBMClassifier(**params)
        model.fit(x_train, y_train)
        preds = model.predict_proba(x_val)[:, 1]
        return float(log_loss(y_val, preds))

    study = optuna.create_study(**_default_optuna_study_kwargs())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_xgboost_regressor(
    x_train,
    y_train,
    x_val,
    y_val,
    n_trials: int = 50,
    random_state: int = 20260510,
) -> Dict[str, Any]:
    require_package("xgboost", "pip install xgboost")
    require_package("numpy", "pip install -r requirements.txt")
    require_package("sklearn", "pip install -r requirements.txt")
    import numpy as np
    from sklearn.metrics import mean_squared_error
    from xgboost import XGBRegressor

    try:
        import optuna
    except Exception:
        return {}

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "random_state": random_state,
            "n_jobs": -1,
        }
        model = XGBRegressor(**params)
        model.fit(x_train, y_train)
        preds = model.predict(x_val)
        return float(np.sqrt(mean_squared_error(y_val, preds)))

    study = optuna.create_study(**_default_optuna_study_kwargs())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_xgboost_classifier(
    x_train,
    y_train,
    x_val,
    y_val,
    n_trials: int = 50,
    random_state: int = 20260510,
) -> Dict[str, Any]:
    require_package("xgboost", "pip install xgboost")
    require_package("numpy", "pip install -r requirements.txt")
    require_package("sklearn", "pip install -r requirements.txt")
    import numpy as np
    from sklearn.metrics import log_loss
    from xgboost import XGBClassifier

    try:
        import optuna
    except Exception:
        return {}

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": random_state,
            "n_jobs": -1,
        }
        model = XGBClassifier(**params)
        model.fit(x_train, y_train)
        preds = model.predict_proba(x_val)[:, 1]
        return float(log_loss(y_val, preds))

    study = optuna.create_study(**_default_optuna_study_kwargs())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ── Stacking Ensemble Models ─────────────────────────────────────────────────


def build_stacking_regressor(
    random_state: int = 20260510,
) -> object:
    require_package("sklearn", "pip install -r requirements.txt")
    from sklearn.ensemble import (
        HistGradientBoostingRegressor,
        RandomForestRegressor,
        StackingRegressor,
    )
    from sklearn.linear_model import Ridge

    base_estimators = [
        ("rf", RandomForestRegressor(
            n_estimators=200, min_samples_leaf=50, max_features="sqrt",
            max_samples=0.7, n_jobs=-1, random_state=random_state,
        )),
        ("hgb", HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, l2_regularization=0.05,
            random_state=random_state,
        )),
    ]
    try:
        from lightgbm import LGBMRegressor
        base_estimators.append(("lightgbm", LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            min_child_samples=80, random_state=random_state,
            n_jobs=-1, verbose=-1,
        )))
    except Exception:
        pass
    try:
        from xgboost import XGBRegressor
        base_estimators.append(("xgboost", XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.85, colsample_bytree=0.85,
            objective="reg:squarederror", tree_method="hist",
            random_state=random_state, n_jobs=-1,
        )))
    except Exception:
        pass

    return StackingRegressor(
        estimators=base_estimators,
        final_estimator=Ridge(alpha=1.0, random_state=random_state),
        cv=5,
        n_jobs=-1,
    )


def build_stacking_classifier(
    random_state: int = 20260510,
) -> object:
    require_package("sklearn", "pip install -r requirements.txt")
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        RandomForestClassifier,
        StackingClassifier,
    )
    from sklearn.linear_model import LogisticRegression

    base_estimators = [
        ("rf", RandomForestClassifier(
            n_estimators=200, min_samples_leaf=50, max_features="sqrt",
            max_samples=0.7, n_jobs=-1, random_state=random_state,
            class_weight="balanced_subsample",
        )),
        ("hgb", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, l2_regularization=0.05,
            random_state=random_state,
        )),
    ]
    try:
        from lightgbm import LGBMClassifier
        base_estimators.append(("lightgbm", LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            min_child_samples=80, random_state=random_state,
            n_jobs=-1, verbose=-1,
        )))
    except Exception:
        pass
    try:
        from xgboost import XGBClassifier
        base_estimators.append(("xgboost", XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.85, colsample_bytree=0.85,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", random_state=random_state, n_jobs=-1,
        )))
    except Exception:
        pass

    return StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(C=1.0, max_iter=3000, n_jobs=-1),
        cv=5,
        n_jobs=-1,
    )
