"""Ordinal QWK using the caller's complete, explicitly ordered class space."""

from collections.abc import Sequence
from math import isfinite
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import cohen_kappa_score


def validate_classes(classes: Sequence[Any]) -> list[Any]:
    values = list(classes)
    if len(values) < 2:
        raise ValueError("At least two ordered classes are required.")
    strings = all(isinstance(value, str) and value for value in values)
    numbers = all(
        isinstance(value, Real)
        and not isinstance(value, (bool, np.bool_))
        and isfinite(float(value))
        for value in values
    )
    if not (strings or numbers):
        raise ValueError("Classes must be all strings or all finite numeric labels.")
    if len(set(values)) != len(values):
        raise ValueError("Classes must be unique and supplied in ordinal order.")
    return values


def encode_labels(labels: ArrayLike, classes: Sequence[Any]) -> NDArray[np.int64]:
    ordered = validate_classes(classes)
    array = np.asarray(labels, dtype=object)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Labels must be a nonempty one-dimensional array.")
    ranks = {label: rank for rank, label in enumerate(ordered)}
    try:
        if any(isinstance(label, (bool, np.bool_)) for label in array):
            raise ValueError("Boolean labels are not supported.")
        return np.asarray([ranks[label] for label in array], dtype=np.int64)
    except (KeyError, TypeError) as error:
        raise ValueError("Labels contain values outside the declared class space.") from error


def quadratic_weighted_kappa(
    y_true: ArrayLike, y_pred: ArrayLike, *, classes: Sequence[Any]
) -> float:
    """Compute rank-distance QWK, including classes absent from this fold.

    Class order is never inferred. Identical constant labels have undefined QWK;
    raise explicitly instead of inventing a perfect score or persisting NaN.
    """
    ordered = validate_classes(classes)
    truth = encode_labels(y_true, ordered)
    prediction = encode_labels(y_pred, ordered)
    if len(truth) != len(prediction):
        raise ValueError("True and predicted labels must have the same length.")
    if np.all(truth == truth[0]) and np.all(prediction == truth[0]):
        raise ValueError("QWK is undefined for identical constant labels.")
    result = float(
        cohen_kappa_score(truth, prediction, labels=list(range(len(ordered))), weights="quadratic")
    )
    if not np.isfinite(result):
        raise ValueError("QWK is undefined for these labels.")
    return result
