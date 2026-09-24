import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from pet_adoption.metrics import quadratic_weighted_kappa


def test_perfect_match():
    result = quadratic_weighted_kappa([0, 1, 2], [0, 1, 2], classes=[0, 1, 2])
    assert result == 1.0
    assert type(result) is float


def test_reversed_order():
    assert quadratic_weighted_kappa([0, 1, 2], [2, 1, 0], classes=[0, 1, 2]) == pytest.approx(-1.0)


def test_partial_errors():
    assert quadratic_weighted_kappa([0, 1, 2], [0, 2, 2], classes=[0, 1, 2]) == pytest.approx(0.8)


def test_absent_class_preserves_full_space():
    truth, prediction, classes = [0, 3, 3], [0, 1, 3], [0, 1, 2, 3]
    assert quadratic_weighted_kappa(truth, prediction, classes=classes) == pytest.approx(
        cohen_kappa_score(truth, prediction, labels=classes, weights="quadratic")
    )
    assert quadratic_weighted_kappa(truth, prediction, classes=classes) != pytest.approx(
        cohen_kappa_score(truth, prediction, weights="quadratic")
    )


def test_arbitrary_ordered_labels():
    assert quadratic_weighted_kappa(
        ["soon", "later", "never"],
        ["never", "later", "soon"],
        classes=["soon", "later", "never"],
    ) == pytest.approx(-1)
    assert quadratic_weighted_kappa(
        [10, 90, -5], [-5, 90, 10], classes=[10, 90, -5]
    ) == pytest.approx(-1)


@pytest.mark.parametrize(
    "truth,prediction,classes",
    [
        ([0, 3], [0, 1], [0, 1, 2]),
        ([0, 1], [0, -1], [0, 1, 2]),
        ([0, np.nan], [0, 1], [0, 1]),
        ([0, 1], [0, 1], [0, 1, 1]),
        ([0, 1], [0, 1], [0, np.inf]),
        ([0, 1], [False, True], [0, 1]),
        ([], [], [0, 1]),
        ([[0, 1]], [[0, 1]], [0, 1]),
    ],
)
def test_invalid_labels(truth, prediction, classes):
    with pytest.raises(ValueError):
        quadratic_weighted_kappa(truth, prediction, classes=classes)


def test_different_lengths():
    with pytest.raises(ValueError, match="same length"):
        quadratic_weighted_kappa([0, 1], [0], classes=[0, 1])


def test_identical_constant_labels_are_explicitly_undefined():
    with pytest.raises(ValueError, match="undefined"):
        quadratic_weighted_kappa([1, 1], [1, 1], classes=[0, 1, 2])
