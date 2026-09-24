"""Fast checks of the pieces that silently break a submission if they regress."""
import numpy as np
import pandas as pd
import pytest

import src.utils as utils
from src.config import ID_COL, TARGET_COL
from src.features import knn_target_features, parse_age_months
from src.utils import OptimizedRounder, qwk


def test_rounder_predicts_labels_1_to_4():
    rng = np.random.default_rng(0)
    y = rng.integers(1, 5, 500)
    x = y + rng.normal(0, 0.5, 500)
    pred = OptimizedRounder().fit(x, y).predict(x)
    assert set(np.unique(pred)) <= {1, 2, 3, 4}
    assert qwk(y, pred) > 0.8


def test_write_submission_restores_raw_ids(tmp_path, monkeypatch):
    raw = pd.DataFrame({ID_COL: ["95314294", "a4ade8bb2"], "Description": ["x", "y"]})
    raw.to_csv(tmp_path / "test.csv", index=False)
    monkeypatch.setattr(utils, "TEST_CSV", tmp_path / "test.csv")
    monkeypatch.setattr(utils, "SUBMISSIONS_DIR", tmp_path)

    # load_data() zero-pads ids to 9 chars; the file must contain them as in test.csv
    path = utils.write_submission(pd.Series(["095314294", "a4ade8bb2"]), np.array([2, 3]), "sub")
    sub = pd.read_csv(path, dtype=str)
    assert sub[ID_COL].tolist() == ["95314294", "a4ade8bb2"]
    assert sub[TARGET_COL].tolist() == ["2", "3"]

    with pytest.raises(ValueError):
        utils.write_submission(pd.Series(["095314294"]), np.array([2]), "sub")


def test_knn_target_is_out_of_fold():
    # Two identical rows in different folds: each must see the other's label, never its own
    X = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    y = np.array([1.0, 4.0, 2.0, 3.0])
    folds = np.array([0, 1, 0, 1])
    oof, _ = knn_target_features(X, X[:1], y, folds, prefix="t", k=1)
    assert oof["t_nn1_y"].tolist() == [4.0, 1.0, 3.0, 2.0]


@pytest.mark.parametrize("text, months", [
    ("2 month old kitten", 2.0),
    ("He is 1 year old", 12.0),
    ("dua bulan", 2.0),
    ("rescued 2 years ago, now 4 months", 4.0),
])
def test_parse_age_months(text, months):
    assert parse_age_months(text) == pytest.approx(months)
