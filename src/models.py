"""Shared CV loop for gradient boosting on tabular features."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import SEED
from src.utils import OptimizedRounder, qwk

LGB_PARAMS = dict(
    objective="regression", learning_rate=0.02, num_leaves=31,
    feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=1.0, min_data_in_leaf=20, verbose=-1, seed=SEED,
)


def ridge_oof(
    X_train, X_test, y: np.ndarray, folds: np.ndarray, alpha: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Level-1 Ridge regression on fixed folds → (OOF predictions, fold-averaged test predictions).

    Works with dense embeddings and sparse TF-IDF alike. The OOF output is used
    as a stacking feature for LightGBM.
    """
    from sklearn.linear_model import Ridge

    y = np.asarray(y, dtype=float)
    oof = np.zeros(X_train.shape[0])
    pred_test = np.zeros(X_test.shape[0])
    fold_ids = np.unique(folds)
    for f in fold_ids:
        tr, va = folds != f, folds == f
        model = Ridge(alpha=alpha).fit(X_train[tr], y[tr])
        oof[va] = model.predict(X_train[va])
        pred_test += model.predict(X_test) / len(fold_ids)
    return oof, pred_test


def run_lgb_cv(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y: np.ndarray,
    folds: np.ndarray,
    params: dict | None = None,
    verbose: bool = True,
) -> dict:
    """LightGBM regression on fixed folds.

    Returns dict with ``oof``, ``test`` (fold-averaged), ``qwk`` (OOF, optimised
    thresholds), ``rounder`` and ``importance`` (mean gain per feature).
    """
    params = {**LGB_PARAMS, **(params or {})}
    y = np.asarray(y, dtype=float)
    oof = np.zeros(len(X_train))
    pred_test = np.zeros(len(X_test))
    importance = np.zeros(X_train.shape[1])
    n_folds = len(np.unique(folds))
    for f in np.unique(folds):
        tr, va = folds != f, folds == f
        model = lgb.train(
            params,
            lgb.Dataset(X_train[tr], y[tr]),
            num_boost_round=5000,
            valid_sets=[lgb.Dataset(X_train[va], y[va])],
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        oof[va] = model.predict(X_train[va], num_iteration=model.best_iteration)
        pred_test += model.predict(X_test, num_iteration=model.best_iteration) / n_folds
        importance += model.feature_importance("gain") / n_folds
        if verbose:
            rmse = np.sqrt(np.mean((oof[va] - y[va]) ** 2))
            print(f"fold {f}: best_iter={model.best_iteration:4d}  rmse={rmse:.4f}")
    rounder = OptimizedRounder().fit(oof, y)
    score = qwk(y, rounder.predict(oof))
    if verbose:
        print(f"OOF QWK: {score:.4f}   thresholds: {rounder.coef_.round(3)}")
    imp = pd.Series(importance, index=X_train.columns).sort_values(ascending=False)
    return {"oof": oof, "test": pred_test, "qwk": score, "rounder": rounder, "importance": imp}
