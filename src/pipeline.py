"""End-to-end pipeline behind ``notebooks/final_submission.ipynb``: raw data → submission.

Each function reproduces one development notebook (02–07) with the parameters
fixed there. Only the expensive, deterministic-input steps are cached in
``cfg.cache_dir`` (image embeddings, DeBERTa predictions); set
``cfg.use_cache = False`` to recompute everything from ``data/raw``.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import DATA_DIR, ID_COL, N_FOLDS, SEED, TARGET_COL, TEXT_COL
from src.data import add_folds, add_photo_count, build_image_index, load_data, text_meta_features
from src.features import (
    char_tfidf, duplicate_groups, group_size, knn_target_features, normalize_text,
    text_regex_features, tfidf_word_char,
)
from src.image_head import HeadConfig, build_bags, run_head_cv
from src.images import (
    CLIP_MODEL, CLIP_PRETRAINED, aggregate_per_pet, encode_texts, encoder_feature_pipeline,
    extract_image_features, load_clip, pet_embeddings, zero_shot_features,
)
from src.models import fit_lgb_full, ridge_oof, run_lgb_cv
from src.text import TextConfig, run_text_cv
from src.utils import OptimizedRounder, qwk


@dataclass
class FinalConfig:
    seed: int = SEED
    n_folds: int = N_FOLDS
    cache_dir: Path = DATA_DIR / "final_cache"
    use_cache: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # 02 · text features
    text_dup_threshold: float = 0.9
    # 03 · image encoders
    clip_model: str = CLIP_MODEL
    clip_pretrained: str = CLIP_PRETRAINED
    clip_logit_scale: float = 100.0
    img_dup_threshold: float = 0.95
    siglip_model: str = "ViT-SO400M-16-SigLIP2-384"
    siglip_pretrained: str = "webli"
    # 04 / 03§6 · level-1 Ridge alphas
    ridge_alphas: dict = field(default_factory=lambda: {
        "ridge_clip_img": 3.3, "ridge_tfidf": 3.0, "ridge_clip_text": 3.3,
        "ridge_siglip_img": 3.0, "ridge_siglip_text": 1.0,
    })
    # 05 · transformer
    text_cfg: TextConfig = field(default_factory=TextConfig)
    # 06 · attention head
    head_cfg: HeadConfig = field(default_factory=lambda: HeadConfig(epochs=7))
    head_seeds: int = 5
    # 07 · level-2
    lgb_params: dict = field(default_factory=lambda: dict(learning_rate=0.01, num_leaves=15, min_data_in_leaf=40))
    lgb_seeds: int = 5
    refit_rounds_mult: float = 1.15


LEVEL1_NAMES = {
    "ridge_clip_img": "Ridge · CLIP фото",
    "ridge_siglip_img": "Ridge · SigLIP2 фото",
    "head_img": "Attention-голова · фото",
    "ridge_tfidf": "Ridge · TF-IDF",
    "ridge_clip_text": "Ridge · CLIP текст",
    "ridge_siglip_text": "Ridge · SigLIP2 текст",
    "text_deberta": "DeBERTa-v3-base",
}


def oof_qwk(y: np.ndarray, pred: np.ndarray) -> float:
    """QWK of continuous OOF predictions after threshold optimisation."""
    return qwk(y, OptimizedRounder().fit(pred, y).predict(pred))


def _free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# 01 · data
# --------------------------------------------------------------------------- #
def prepare_data(cfg: FinalConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Raw csv + image folder → (train with ``fold`` and ``n_photos``, test, image index)."""
    train, test = load_data()
    img_index = build_image_index()
    train = add_folds(add_photo_count(train, img_index), n_folds=cfg.n_folds, seed=cfg.seed)
    test = add_photo_count(test, img_index)
    return train, test, img_index


# --------------------------------------------------------------------------- #
# 02 · text features
# --------------------------------------------------------------------------- #
def text_features(train: pd.DataFrame, test: pd.DataFrame, cfg: FinalConfig) -> pd.DataFrame:
    """Meta + regex features, OOF kNN-target on char TF-IDF, duplicate-group size (train rows first)."""
    n_tr = len(train)
    all_text = pd.concat([train[TEXT_COL], test[TEXT_COL]], ignore_index=True)
    feats = pd.concat([text_meta_features(all_text), text_regex_features(all_text)], axis=1)

    X_char = char_tfidf(all_text)
    valid = (normalize_text(all_text).str.len() >= 5).to_numpy()
    knn_tr, knn_te = knn_target_features(
        X_char[:n_tr], X_char[n_tr:], train[TARGET_COL].to_numpy(float), train["fold"].to_numpy(),
        prefix="txt", train_valid=valid[:n_tr], test_valid=valid[n_tr:],
    )
    feats = pd.concat([feats, pd.concat([knn_tr, knn_te], ignore_index=True)], axis=1)
    feats["txt_dup_size"] = group_size(duplicate_groups(X_char, valid, threshold=cfg.text_dup_threshold))
    return feats


# --------------------------------------------------------------------------- #
# 03 · image features
# --------------------------------------------------------------------------- #
def clip_features(
    img_index: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, cfg: FinalConfig
) -> tuple[pd.DataFrame, dict]:
    """CLIP pass (embeddings + photo quality) → per-pet zero-shot/quality aggregates, OOF kNN-target,
    text↔image similarity, duplicate-group size. Returns (features, embeddings for level-1 models)."""
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    emb_path, qual_path = cfg.cache_dir / "clip_img_emb.npy", cfg.cache_dir / "clip_img_quality.parquet"
    model, preprocess, tokenizer = load_clip(cfg.clip_model, cfg.clip_pretrained, cfg.device)
    if cfg.use_cache and emb_path.exists() and qual_path.exists():
        img_emb, img_qual = np.load(emb_path), pd.read_parquet(qual_path)
    else:
        img_emb, img_qual = extract_image_features(
            img_index["path"].tolist(), model, preprocess, cfg.device, batch_size=128, num_workers=6
        )
        np.save(emb_path, img_emb)
        img_qual.to_parquet(qual_path, index=False)

    zs = zero_shot_features(img_emb, model, tokenizer, cfg.device, logit_scale=cfg.clip_logit_scale)
    pet_img = aggregate_per_pet(img_index, pd.concat([img_qual, zs], axis=1), aggs=("first", "mean", "max"))

    pet_ids, emb_first, emb_mean = pet_embeddings(img_index, img_emb)
    pos = pd.Series(np.arange(len(pet_ids)), index=pet_ids)
    tr_pos, te_pos = pos[train[ID_COL]].to_numpy(), pos[test[ID_COL]].to_numpy()
    E_tr, E_te = emb_mean[tr_pos], emb_mean[te_pos]

    y = train[TARGET_COL].to_numpy(float)
    knn_tr, knn_te = knn_target_features(E_tr, E_te, y, train["fold"].to_numpy(), prefix="img")
    T_tr = encode_texts(train[TEXT_COL].str.slice(0, 300).tolist(), model, tokenizer, cfg.device)
    T_te = encode_texts(test[TEXT_COL].str.slice(0, 300).tolist(), model, tokenizer, cfg.device)
    img_dup_size = group_size(duplicate_groups(np.vstack([E_tr, E_te]), threshold=cfg.img_dup_threshold))
    del model
    _free_gpu()

    feats = pd.concat([train[[ID_COL]], test[[ID_COL]]], ignore_index=True).merge(pet_img, on=ID_COL, how="left")
    feats[knn_tr.columns] = pd.concat([knn_tr, knn_te], ignore_index=True).to_numpy()
    T = np.vstack([T_tr, T_te])
    feats["clip_txt_img_mean"] = (T * np.vstack([E_tr, E_te])).sum(1)
    feats["clip_txt_img_first"] = (T * np.vstack([emb_first[tr_pos], emb_first[te_pos]])).sum(1)
    feats["img_dup_size"] = img_dup_size
    embs = {
        "img_emb": img_emb,
        "pet_img": np.vstack([np.hstack([E_tr, emb_first[tr_pos]]), np.hstack([E_te, emb_first[te_pos]])]),
        "text": T,
    }
    return feats.drop(columns=ID_COL), embs


def siglip_features(
    img_index: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, cfg: FinalConfig
) -> tuple[pd.DataFrame, dict]:
    """Same pipeline for SigLIP2 (prefixed ``siglip_``). Returns (features, embeddings)."""
    sig_tr, sig_te = encoder_feature_pipeline(
        "siglip", cfg.siglip_model, cfg.siglip_pretrained, img_index, train, test, cfg.cache_dir,
        cfg.device, use_cache=cfg.use_cache,
    )
    _free_gpu()
    load = lambda name: np.vstack([np.load(cfg.cache_dir / f"siglip_{name}_{s}.npy") for s in ("train", "test")])  # noqa: E731
    embs = {
        "img_emb": np.load(cfg.cache_dir / "siglip_img_emb.npy"),
        "pet_img": np.hstack([load("pet_mean"), load("pet_first")]),
        "text": load("text"),
    }
    feats = pd.concat([sig_tr, sig_te], ignore_index=True).drop(columns=ID_COL)
    return feats, embs


# --------------------------------------------------------------------------- #
# 04–06 · level-1 models
# --------------------------------------------------------------------------- #
def ridge_models(
    train: pd.DataFrame, test: pd.DataFrame, clip_embs: dict, siglip_embs: dict, cfg: FinalConfig
) -> pd.DataFrame:
    """Five Ridge level-1 models → OOF (train rows) + fold-averaged test predictions."""
    n_tr = len(train)
    y, folds = train[TARGET_COL].to_numpy(float), train["fold"].to_numpy()
    tfidf = tfidf_word_char(pd.concat([train[TEXT_COL], test[TEXT_COL]], ignore_index=True))
    inputs = {
        "ridge_clip_img": clip_embs["pet_img"], "ridge_tfidf": tfidf, "ridge_clip_text": clip_embs["text"],
        "ridge_siglip_img": siglip_embs["pet_img"], "ridge_siglip_text": siglip_embs["text"],
    }
    out = {}
    for name, X in inputs.items():
        oof, pred_te = ridge_oof(X[:n_tr], X[n_tr:], y, folds, alpha=cfg.ridge_alphas[name])
        out[name] = np.r_[oof, pred_te]
    return pd.DataFrame(out)


def deberta_model(train: pd.DataFrame, test: pd.DataFrame, cfg: FinalConfig) -> np.ndarray:
    """Fine-tuned DeBERTa-v3-base on the folds (~18 min on a 12 GB GPU); cached."""
    path = cfg.cache_dir / "deberta_pred.npy"
    if cfg.use_cache and path.exists():
        return np.load(path)
    oof, pred_te, _ = run_text_cv(train, test, cfg.text_cfg, device=cfg.device)
    pred = np.r_[oof, pred_te]
    np.save(path, pred)
    _free_gpu()
    return pred


def attention_head_model(
    img_index: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame,
    clip_embs: dict, siglip_embs: dict, cfg: FinalConfig,
) -> np.ndarray:
    """Gated-attention head over CLIP ⊕ SigLIP2 photo embeddings, averaged over seeds."""
    embs = [clip_embs["img_emb"], siglip_embs["img_emb"]]
    bags_tr, mask_tr = build_bags(img_index, embs, train[ID_COL], cfg.head_cfg.max_photos)
    bags_te, mask_te = build_bags(img_index, embs, test[ID_COL], cfg.head_cfg.max_photos)
    y, folds = train[TARGET_COL].to_numpy(float), train["fold"].to_numpy()
    oof, pred_te = np.zeros(len(train)), np.zeros(len(test))
    for s in range(cfg.head_seeds):
        head_cfg = HeadConfig(**{**cfg.head_cfg.__dict__, "seed": cfg.seed + s})
        res = run_head_cv(bags_tr, mask_tr, bags_te, mask_te, y, folds, head_cfg, device=cfg.device, verbose=False)
        oof += res["oof"] / cfg.head_seeds
        pred_te += res["test"] / cfg.head_seeds
    del bags_tr, bags_te
    _free_gpu()
    return np.r_[oof, pred_te]


# --------------------------------------------------------------------------- #
# 07 · level-2 stack, thresholds, full refit
# --------------------------------------------------------------------------- #
def level2_stack(X: pd.DataFrame, train: pd.DataFrame, cfg: FinalConfig) -> dict:
    """LightGBM on features + level-1 predictions, several seeds; thresholds from OOF;
    honest QWK with thresholds fit on the other folds; refit on all train rows."""
    n_tr = len(train)
    y, folds = train[TARGET_COL].to_numpy(float), train["fold"].to_numpy()
    X_tr, X_te = X[:n_tr], X[n_tr:]
    seeds = [cfg.seed + s for s in range(cfg.lgb_seeds)]

    oof, pred_te_cv, best_iters, seed_scores = np.zeros(n_tr), np.zeros(len(X_te)), [], []
    importance = 0
    for s in seeds:
        res = run_lgb_cv(X_tr, X_te, y, folds, params={**cfg.lgb_params, "seed": s}, verbose=False)
        oof += res["oof"] / len(seeds)
        pred_te_cv += res["test"] / len(seeds)
        importance = importance + res["importance"] / len(seeds)
        best_iters += res["best_iters"]
        seed_scores.append(res["qwk"])
    rounder = OptimizedRounder().fit(oof, y)

    nested = np.zeros(n_tr)
    for f in np.unique(folds):
        nested[folds == f] = OptimizedRounder().fit(oof[folds != f], y[folds != f]).predict(oof[folds == f])

    n_rounds = int(np.mean(best_iters) * cfg.refit_rounds_mult)
    pred_te = np.zeros(len(X_te))
    for s in seeds:
        pred_te += fit_lgb_full(X_tr, y, n_rounds, params={**cfg.lgb_params, "seed": s}).predict(X_te) / len(seeds)

    return {
        "oof": oof, "test": pred_te, "test_cv": pred_te_cv, "rounder": rounder,
        "qwk": qwk(y, rounder.predict(oof)), "qwk_honest": qwk(y, nested), "seed_qwk": seed_scores,
        "n_rounds": n_rounds, "importance": importance,
    }
