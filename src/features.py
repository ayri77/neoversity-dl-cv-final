"""Hand-crafted text features, near-duplicate groups and OOF kNN-target features.

The competition data has no tabular columns, so we try to recover the most
useful ones from the original PetFinder data (type, age, breed, health,
rescuer activity) out of free text. Near-duplicate listings (same rescuer,
same template) turn out to share almost the same target, which is exploited
through out-of-fold nearest-neighbour target features.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, spmatrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

NUM_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "satu": 1, "dua": 2, "tiga": 3, "empat": 4, "lima": 5, "enam": 6,
}
_NUM = r"(\d+(?:\.\d+)?|" + "|".join(NUM_WORDS) + r"|half)"

# Unit → months multiplier; Malay units included (bulan = month, tahun = year, minggu = week)
_AGE_UNITS = {
    r"(?:weeks?|wks?|minggu)": 12 / 52,
    r"(?:months?|mths?|mos?|bulan)": 1.0,
    r"(?:years?|yrs?|tahun)": 12.0,
}

# Keyword groups: feature name → regex (applied to lower-cased text)
KEYWORDS = {
    "kw_cat": r"\b(?:cats?|kittens?|kitty|kitties|meow|kucing)\b",
    "kw_dog": r"\b(?:dogs?|pupp(?:y|ies)|pups?|doggy|anjing)\b",
    "kw_baby": r"\b(?:kittens?|pupp(?:y|ies)|pups?|babies|baby|newborn)\b",
    "kw_adult": r"\b(?:adult|senior|old (?:cat|dog)|elderly|mother cat|mama)\b",
    "kw_male": r"\b(?:male|boy|he|him|his|jantan)\b",
    "kw_female": r"\b(?:female|girl|she|her|betina)\b",
    "kw_pure": (
        r"persian|siamese|british short ?hair|maine coon|bengal|ragdoll|scottish fold|"
        r"husky|poodle|shih ?tzu|golden retriever|labrador|rottweiler|german shepherd|"
        r"beagle|chihuahua|pomeranian|dachshund|terrier|schnauzer|corgi|pug|spitz|"
        r"pure ?bre(?:e)?d|pedigree"
    ),
    "kw_mixed": r"\bmix(?:ed)?\b|cross ?breed|\blocal\b|domestic (?:short|medium|long)|\bdsh\b|kampung",
    "kw_vaccinated": r"vaccin|vacinat|jab|injection",
    "kw_dewormed": r"de-?worm",
    "kw_sterilized": r"spay|neuter|steril|castrat|desex",
    "kw_healthy": r"\bhealthy\b|\bhealth\b",
    "kw_sick": r"\bsick|injur|wound|blind|lame|limp|disease|virus|parvo|skin problem|fractur|paraly|surgery|vet",
    "kw_trained": r"toilet|litter|potty|house ?trained|obedien",
    "kw_friendly": r"friendly|playful|active|manja|affection|cute|adorable|lovely|sweet|gentle",
    "kw_rescued": r"rescu|abandon|stray|dump|found|street",
    "kw_fee": r"\bfee\b|\brm ?\d|\bcharge",
    "kw_free": r"\bfree\b|percuma",
    "kw_urgent": r"urgent|asap|immediately|pls|please",
    "kw_contact": r"call|whatsapp|\bwa\b|sms|contact|email|\bpm\b|text me",
    "kw_group": r"\b(?:them|siblings|litter|brothers|sisters|each|both|all of them)\b",
}
_KEYWORDS_RE = {k: re.compile(v) for k, v in KEYWORDS.items()}
_CJK_RE = re.compile(r"[一-鿿]")
_MALAY_RE = re.compile(r"\b(?:dan|untuk|yang|anak|ini|saya|ada|tak|nak|sangat|comel|dengan)\b")
_COUNT_RE = re.compile(
    _NUM + r"\s+(?:\w+\s+)?(?:kittens|puppies|pups|cats|dogs|siblings|babies|of them)\b"
)


def _to_number(tok: str) -> float:
    if tok == "half":
        return 0.5
    return float(NUM_WORDS.get(tok, tok))


def parse_age_months(text: str) -> float:
    """Return the smallest age mentioned in the text, in months (NaN if none).

    The minimum is used because descriptions often mention both the pet's age
    and irrelevant durations ("rescued 2 years ago"), and small values are the
    most informative (kittens/puppies are adopted fastest).
    """
    t = text.lower()
    ages = []
    for unit, mult in _AGE_UNITS.items():
        for m in re.finditer(_NUM + r"\s*(?:-|\+|and a half)?\s*" + unit + r"\b", t):
            ages.append(_to_number(m.group(1)) * mult)
    ages = [a for a in ages if 0 < a <= 240]
    return min(ages) if ages else np.nan


def parse_pet_count(text: str) -> float:
    """Largest explicit count of animals mentioned ("3 kittens", "two puppies")."""
    counts = [_to_number(m.group(1)) for m in _COUNT_RE.finditer(text.lower())]
    counts = [c for c in counts if 1 <= c <= 20]
    return max(counts) if counts else np.nan


def text_regex_features(text: pd.Series) -> pd.DataFrame:
    """Keyword flags plus parsed age / pet count / language indicators."""
    s = text.fillna("").astype(str)
    low = s.str.lower()
    feats = pd.DataFrame(index=s.index)
    for name, rx in _KEYWORDS_RE.items():
        feats[name] = low.apply(lambda t, rx=rx: len(rx.findall(t)))

    feats["type_text"] = np.sign(feats["kw_dog"] - feats["kw_cat"])  # +1 dog, -1 cat, 0 unknown
    feats["age_months"] = s.apply(parse_age_months)
    feats["age_missing"] = feats["age_months"].isna().astype(int)
    feats["pet_count"] = s.apply(parse_pet_count)
    # Phone numbers and line breaks were stripped by the organisers, so no features for them
    feats["cjk_ratio"] = s.apply(lambda t: len(_CJK_RE.findall(t)) / max(len(t), 1))
    feats["malay_words"] = low.apply(lambda t: len(_MALAY_RE.findall(t)))
    feats["n_sentences"] = s.str.count(r"[.!?]+") + 1
    return feats


# --------------------------------------------------------------------------- #
# Near-duplicate groups and kNN-target features
# --------------------------------------------------------------------------- #
def normalize_text(text: pd.Series) -> pd.Series:
    return text.fillna("").astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def char_tfidf(text: pd.Series, min_df: int = 2) -> spmatrix:
    """L2-normalised char n-gram TF-IDF, used to measure near-duplicate similarity."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=min_df, sublinear_tf=True)
    return vec.fit_transform(normalize_text(text))


def tfidf_word_char(text: pd.Series) -> spmatrix:
    """Word (1-2 gram) + char_wb (2-5 gram) TF-IDF stacked horizontally (CSR)."""
    from scipy.sparse import hstack

    word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50_000, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=5, max_features=80_000, sublinear_tf=True)
    return hstack([word.fit_transform(text), char.fit_transform(text)]).tocsr()


def duplicate_groups(
    X, valid: np.ndarray | None = None, threshold: float = 0.9, n_neighbors: int = 20
) -> np.ndarray:
    """Connected components of the "cosine similarity >= threshold" graph.

    ``X`` is an L2-normalised matrix (sparse TF-IDF or dense embeddings).
    Rows with ``valid == False`` (e.g. empty descriptions) become singletons.
    Returns an integer group id per row.
    """
    n = X.shape[0]
    valid = np.ones(n, bool) if valid is None else np.asarray(valid)
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, n), metric="cosine").fit(X)
    dist, idx = nn.kneighbors(X)
    mask = (1 - dist) >= threshold
    rows = np.repeat(np.arange(n), idx.shape[1])[mask.ravel()]
    cols = idx.ravel()[mask.ravel()]
    keep = valid[rows] & valid[cols]
    adj = coo_matrix((np.ones(keep.sum()), (rows[keep], cols[keep])), shape=(n, n))
    _, labels = connected_components(adj, directed=False)
    return labels


def knn_target_features(
    X_train,
    X_test,
    y: np.ndarray,
    folds: np.ndarray,
    prefix: str,
    k: int = 5,
    power: float = 4.0,
    train_valid: np.ndarray | None = None,
    test_valid: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-fold nearest-neighbour target features.

    For a train row in fold ``f`` neighbours are searched only among train rows
    of the other folds, so the feature never sees the row's own label and the
    OOF score stays honest. Test rows use all train rows. With random stratified
    folds this mimics the test situation (~11% of test pets have a near-duplicate
    in train).

    Features: similarity-weighted mean target of top-k neighbours (weights =
    sim**power, so near-duplicates dominate), target of the nearest neighbour
    and the top similarity itself.
    """
    y = np.asarray(y, dtype=float)
    folds = np.asarray(folds)
    n_tr = X_train.shape[0]
    train_valid = np.ones(n_tr, bool) if train_valid is None else np.asarray(train_valid)
    test_valid = np.ones(X_test.shape[0], bool) if test_valid is None else np.asarray(test_valid)

    def _query(X_ref, y_ref, X_q, q_valid):
        nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(X_ref)
        dist, idx = nn.kneighbors(X_q)
        sim = np.clip(1 - dist, 0, 1)
        ny = y_ref[idx]
        w = sim**power
        out = pd.DataFrame(
            {
                f"{prefix}_knn_y": (w * ny).sum(1) / (w.sum(1) + 1e-9),
                f"{prefix}_nn1_y": ny[:, 0],
                f"{prefix}_nn1_sim": sim[:, 0],
                f"{prefix}_knn_mean_sim": sim.mean(1),
            }
        )
        # Rows without usable content (empty text) get NaN — LightGBM handles it
        out.loc[~q_valid, [f"{prefix}_knn_y", f"{prefix}_nn1_y"]] = np.nan
        return out

    parts = []
    for f in np.unique(folds):
        va = folds == f
        ref = (~va) & train_valid
        part = _query(X_train[ref], y[ref], X_train[va], train_valid[va])
        part.index = np.where(va)[0]
        parts.append(part)
    oof = pd.concat(parts).sort_index()

    test = _query(X_train[train_valid], y[train_valid], X_test, test_valid)
    return oof, test


def group_size(groups: np.ndarray) -> np.ndarray:
    """Size of each row's group over train+test (proxy for rescuer activity; no labels used)."""
    g = pd.Series(groups)
    return g.map(g.value_counts()).to_numpy()
