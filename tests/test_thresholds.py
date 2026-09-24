import numpy as np
import pytest

from pet_adoption.metrics import quadratic_weighted_kappa
from pet_adoption.thresholds import apply_thresholds, fit_thresholds, validate_cutpoints


def test_boundary_and_arbitrary_class_order():
    assert apply_thresholds([-1, 0, 1, 2], [0, 2], classes=["slow", "middle", "fast"]).tolist() == [
        "slow",
        "middle",
        "middle",
        "fast",
    ]


@pytest.mark.parametrize("cutpoints", [[1, 1], [2, 1], [1], [np.nan, 2], [1, np.inf], [[1, 2]]])
def test_invalid_cutpoints(cutpoints):
    with pytest.raises(ValueError):
        validate_cutpoints(cutpoints, n_classes=3)


@pytest.mark.parametrize("scores", [[], [np.inf], [np.nan], [[1]]])
def test_invalid_scores(scores):
    with pytest.raises(ValueError):
        apply_thresholds(scores, [1], classes=[0, 1])


@pytest.mark.parametrize("classes", [[10, 20], ["a", "c", "b"], list(range(7))])
def test_optimizer_is_ordered_reproducible_and_improves(classes):
    scores = np.repeat(np.arange(len(classes), dtype=float), 4)
    truth = np.repeat(classes, 4)
    initial = np.linspace(-2, -1, len(classes) - 1)
    before = quadratic_weighted_kappa(
        truth, apply_thresholds(scores, initial, classes=classes), classes=classes
    )
    fit = fit_thresholds(truth, scores, classes=classes, initial_cutpoints=initial)
    assert fit.calibration_qwk > before
    assert fit.calibration_qwk == pytest.approx(
        quadratic_weighted_kappa(
            truth, apply_thresholds(scores, fit.cutpoints, classes=classes), classes=classes
        )
    )
    assert np.all(np.diff(fit.cutpoints) > 0)
    assert fit == fit_thresholds(truth, scores, classes=classes, initial_cutpoints=initial)


def test_optimizer_rejects_bad_initial_order():
    with pytest.raises(ValueError, match="strictly increasing"):
        fit_thresholds([0, 1, 2], [0, 1, 2], classes=[0, 1, 2], initial_cutpoints=[1, 0])


def test_default_initialization_solves_separated_seven_classes():
    classes = list(range(7))
    fit = fit_thresholds(classes, classes, classes=classes)
    assert fit.calibration_qwk == pytest.approx(1.0)


def test_missing_target_class_supported():
    fit = fit_thresholds([0, 0, 3, 3], [0, 0.2, 2.8, 3], classes=[0, 1, 2, 3])
    assert len(fit.cutpoints) == 3
    assert fit.calibration_qwk == 1.0


def test_constant_targets_are_rejected():
    with pytest.raises(ValueError, match="observed target"):
        fit_thresholds([1, 1], [0, 1], classes=[0, 1])


def test_constant_scores_require_explicit_initialization():
    with pytest.raises(ValueError, match="Constant scores"):
        fit_thresholds([0, 1], [1, 1], classes=[0, 1])
    fit = fit_thresholds([0, 1], [1, 1], classes=[0, 1], initial_cutpoints=[0.5])
    assert fit.calibration_qwk == 0.0


def test_optimizer_length_validation():
    with pytest.raises(ValueError, match="same length"):
        fit_thresholds([0, 1], [0], classes=[0, 1])
