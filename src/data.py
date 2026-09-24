"""Data loading, image indexing, fold assignment and simple meta-features."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.config import (
    ID_COL,
    IMAGES_DIR,
    N_FOLDS,
    SEED,
    TARGET_COL,
    TEST_CSV,
    TEXT_COL,
    TRAIN_CSV,
)

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
PET_ID_LEN = 9


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test csv; fill empty descriptions with an empty string.

    PetIDs are read as strings. Some ids in the provided test.csv lost their
    leading zero (e.g. ``95314294`` vs image ``095314294-1.jpg``), so they are
    left-padded back to the canonical 9-char length.
    """
    train = pd.read_csv(TRAIN_CSV, dtype={ID_COL: str})
    test = pd.read_csv(TEST_CSV, dtype={ID_COL: str})
    for df in (train, test):
        df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
        df[ID_COL] = df[ID_COL].str.zfill(PET_ID_LEN)
    return train, test


def build_image_index() -> pd.DataFrame:
    """Scan the images folder recursively → DataFrame(PetID, path, img_idx, split).

    Files are expected to be named like ``<PetID>-<n>.jpg``; ``split`` is the
    name of the parent folder (``train`` / ``test``).
    """
    rows = []
    for p in IMAGES_DIR.rglob("*"):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        m = re.match(r"^([0-9a-f]+)-(\d+)$", p.stem)
        if not m:
            continue
        rows.append(
            {ID_COL: m.group(1), "path": str(p), "img_idx": int(m.group(2)), "split": p.parent.name}
        )
    return pd.DataFrame(rows).sort_values([ID_COL, "img_idx"]).reset_index(drop=True)


def add_photo_count(df: pd.DataFrame, image_index: pd.DataFrame) -> pd.DataFrame:
    """Attach number of photos per pet (PhotoAmt proxy)."""
    counts = image_index.groupby(ID_COL).size().rename("n_photos")
    out = df.merge(counts, on=ID_COL, how="left")
    out["n_photos"] = out["n_photos"].fillna(0).astype(int)
    return out


def add_folds(df: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = SEED) -> pd.DataFrame:
    """Stratified folds on the target; stored in a ``fold`` column."""
    df = df.copy()
    df["fold"] = -1
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for f, (_, val_idx) in enumerate(skf.split(df, df[TARGET_COL])):
        df.loc[val_idx, "fold"] = f
    return df


def text_meta_features(text: pd.Series) -> pd.DataFrame:
    """Cheap hand-crafted text statistics."""
    s = text.fillna("").astype(str)
    feats = pd.DataFrame(index=s.index)
    feats["desc_len"] = s.str.len()
    feats["desc_words"] = s.str.split().str.len()
    feats["desc_upper_ratio"] = s.apply(
        lambda t: sum(c.isupper() for c in t) / max(len(t), 1)
    )
    feats["desc_excl"] = s.str.count("!")
    feats["desc_digits"] = s.str.count(r"\d")
    feats["desc_is_empty"] = (s.str.strip() == "").astype(int)
    # Non-ASCII share → proxy for Malay/Chinese descriptions
    feats["desc_non_ascii"] = s.apply(
        lambda t: sum(ord(c) > 127 for c in t) / max(len(t), 1)
    )
    return feats
