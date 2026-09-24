"""Shared utilities: reproducibility, metric, threshold optimisation, I/O."""
from __future__ import annotations

import os
import random
from functools import partial

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score

from src.config import ID_COL, SUBMISSIONS_DIR, TARGET_COL


def seed_everything(seed: int = 42) -> None:
    """Fix all random seeds (python, numpy, torch if available)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def qwk(y_true, y_pred) -> float:
    """Quadratic weighted kappa — the official competition metric."""
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


class OptimizedRounder:
    """Convert continuous regression outputs into ordinal classes.

    Learns cut-points that maximise QWK on out-of-fold predictions.
    Classic trick from the original PetFinder competition: predicting a
    continuous score and tuning thresholds beats direct classification.
    """

    def __init__(self, labels: tuple[int, ...] = (1, 2, 3, 4), initial_coef: list[float] | None = None):
        # Labels are kept as-is (1..4 in this competition) — no shifting needed
        self.labels = np.asarray(labels)
        self.coef_: np.ndarray | None = None
        # Default cut-points sit between consecutive labels
        self._initial = initial_coef or list((self.labels[:-1] + self.labels[1:]) / 2)

    def _apply(self, coef: np.ndarray, x: np.ndarray) -> np.ndarray:
        return self.labels[np.digitize(x, np.sort(coef))]

    def _loss(self, coef: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        return -qwk(y, self._apply(coef, x))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "OptimizedRounder":
        loss = partial(self._loss, x=np.asarray(x), y=np.asarray(y))
        res = minimize(loss, self._initial, method="nelder-mead")
        self.coef_ = np.sort(res.x)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Call fit() first")
        return self._apply(self.coef_, np.asarray(x))


def write_submission(ids: pd.Series, preds: np.ndarray, name: str) -> str:
    """Save predictions in the competition format and return the path."""
    sub = pd.DataFrame({ID_COL: ids, TARGET_COL: preds.astype(int)})
    path = SUBMISSIONS_DIR / f"{name}.csv"
    sub.to_csv(path, index=False)
    print(f"Saved {path}  shape={sub.shape}")
    print(sub[TARGET_COL].value_counts().sort_index().to_dict())
    return str(path)
