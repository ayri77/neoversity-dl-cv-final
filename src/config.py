"""Project-wide paths and constants.

All notebooks import paths from here so they don't depend on the
working directory or on where the raw data lives.
"""
from pathlib import Path

# Resolve project root relative to this file: <root>/src/config.py
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
IMAGES_DIR = RAW_DIR / "images"
MODELS_DIR = ROOT_DIR / "models"
SUBMISSIONS_DIR = ROOT_DIR / "submissions"

TRAIN_CSV = RAW_DIR / "train.csv"
TEST_CSV = RAW_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = RAW_DIR / "sample_submission.csv"

ID_COL = "PetID"
TEXT_COL = "Description"
TARGET_COL = "AdoptionSpeed"

SEED = 42
N_FOLDS = 5

for _d in (PROCESSED_DIR, MODELS_DIR, SUBMISSIONS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
