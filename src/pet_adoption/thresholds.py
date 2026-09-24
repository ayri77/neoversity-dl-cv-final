"""Threshold application and bounded calibration, separate from evaluation."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pet_adoption.metrics import encode_labels, quadratic_weighted_kappa, validate_classes


def _scores(scores: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Scores must be a nonempty finite one-dimensional array.")
    return values


def validate_cutpoints(cutpoints: ArrayLike, *, n_classes: int) -> NDArray[np.float64]:
    values = np.asarray(cutpoints, dtype=np.float64)
    if n_classes < 2 or values.ndim != 1 or len(values) != n_classes - 1:
        raise ValueError("Exactly n_classes - 1 cutpoints are required.")
    if not np.isfinite(values).all() or not np.all(values[1:] > values[:-1]):
        raise ValueError("Cutpoints must be finite and strictly increasing.")
    return values.copy()


def apply_thresholds(
    scores: ArrayLike, cutpoints: ArrayLike, *, classes: Sequence[Any]
) -> NDArray[Any]:
    """A score equal to a cutpoint belongs to the higher class."""
    ordered = validate_classes(classes)
    cuts = validate_cutpoints(cutpoints, n_classes=len(ordered))
    return np.asarray(ordered)[np.searchsorted(cuts, _scores(scores), side="right")]


@dataclass(frozen=True)
class ThresholdFit:
    cutpoints: tuple[float, ...]
    calibration_qwk: float
    passes: int
    converged: bool
    method: str = "ordered-coordinate-search-v1"


def fit_thresholds(
    y_true: ArrayLike,
    scores: ArrayLike,
    *,
    classes: Sequence[Any],
    initial_cutpoints: ArrayLike | None = None,
    max_candidates: int = 128,
    max_passes: int = 10,
) -> ThresholdFit:
    """Greedily optimize calibration QWK on an ordered candidate grid.

    Each coordinate only visits points strictly between its neighbors. There is
    no post-fit sorting. This bounded deterministic search is not a global
    optimizer; its in-sample score is not honest model-selection evidence.
    """
    ordered = validate_classes(classes)
    values = _scores(scores)
    truth = encode_labels(y_true, ordered)
    if len(truth) != len(values):
        raise ValueError("Labels and scores must have the same length.")
    if len(np.unique(truth)) < 2:
        raise ValueError("Threshold fitting requires at least two observed target classes.")
    if max_candidates < 2 or max_passes < 1:
        raise ValueError("max_candidates must be >= 2 and max_passes must be >= 1.")
    if initial_cutpoints is None:
        low, high = float(values.min()), float(values.max())
        if low == high:
            raise ValueError("Constant scores require explicit initial_cutpoints.")
        initial_cutpoints = np.linspace(low, high, len(ordered) + 1)[1:-1]
    cuts = validate_cutpoints(initial_cutpoints, n_classes=len(ordered))
    unique = np.unique(values)
    indices = np.linspace(0, len(unique) - 1, min(max_candidates, len(unique))).astype(int)
    # Observed score values cover every distinct assignment under side='right'.
    upper = np.nextafter(unique[-1], np.inf)
    grid = np.unique(np.concatenate((unique[indices], cuts, [upper])))
    grid = grid[np.isfinite(grid)]

    def evaluate(candidate: NDArray[np.float64]) -> float:
        return quadratic_weighted_kappa(
            y_true, apply_thresholds(values, candidate, classes=ordered), classes=ordered
        )

    best = evaluate(cuts)
    for pass_index in range(1, max_passes + 1):
        improved = False
        for index in range(len(cuts)):
            lower = cuts[index - 1] if index else -np.inf
            upper_bound = cuts[index + 1] if index + 1 < len(cuts) else np.inf
            selected = cuts[index]
            for candidate in grid[(grid > lower) & (grid < upper_bound)]:
                trial = cuts.copy()
                trial[index] = candidate
                score = evaluate(trial)
                if score > best:
                    best, selected, improved = score, candidate, True
            cuts[index] = selected
        if not improved:
            return ThresholdFit(tuple(float(value) for value in cuts), best, pass_index, True)
    return ThresholdFit(tuple(float(value) for value in cuts), best, max_passes, False)
