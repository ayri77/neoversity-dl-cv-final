"""Attention-pooling head over the frozen embeddings of all photos of a pet.

Each pet is a *bag* of photos (multiple-instance learning). Every photo is
represented by concatenated frozen embeddings (CLIP ViT-L/14 + SigLIP2) and a
learned position embedding (the first photo is the profile picture). A gated
attention pooling (Ilse et al., 2018, "Attention-based Deep Multiple Instance
Learning") learns which photos matter, instead of the fixed mean / first
aggregation used for the LightGBM features. Optionally the description
embeddings (CLIP text + SigLIP2 text) are added as one extra token.

Everything fits in GPU memory (8.3k pets × 12 photos × 1920 dims in fp16 is
~0.4 GB), so batches are sliced from GPU tensors without a DataLoader.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.config import ID_COL
from src.utils import OptimizedRounder, qwk, seed_everything


@dataclass
class HeadConfig:
    max_photos: int = 12
    hidden: int = 256
    dropout: float = 0.3
    photo_dropout: float = 0.2  # train-time: randomly hide photos (bag augmentation)
    use_text: bool = False
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-2
    seed: int = 42


def build_bags(
    image_index: pd.DataFrame, img_embs: list[np.ndarray], pet_ids: pd.Series, max_photos: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad per-image embeddings into (n_pets, max_photos, dim) + boolean mask, in ``pet_ids`` order.

    ``image_index`` is sorted by (PetID, img_idx) and aligned with every array in ``img_embs``;
    photos beyond ``max_photos`` are dropped (they cover < 5% of pets).
    """
    emb = np.hstack([e.astype(np.float16) for e in img_embs])
    ids = image_index[ID_COL].to_numpy()
    uniq, start, counts = np.unique(ids, return_index=True, return_counts=True)
    pos = pd.Series(np.arange(len(uniq)), index=uniq)[pet_ids].to_numpy()

    bags = np.zeros((len(pet_ids), max_photos, emb.shape[1]), dtype=np.float16)
    mask = np.zeros((len(pet_ids), max_photos), dtype=bool)
    for row, p in enumerate(pos):
        k = min(counts[p], max_photos)
        bags[row, :k] = emb[start[p] : start[p] + k]
        mask[row, :k] = True
    return torch.from_numpy(bags), torch.from_numpy(mask)


class GatedAttentionHead(nn.Module):
    """Photo tokens (+ optional text token) → gated attention pooling → regression."""

    def __init__(self, dim: int, cfg: HeadConfig, target_mean: float, text_dim: int = 0):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, cfg.hidden), nn.GELU(), nn.Dropout(cfg.dropout))
        self.pos = nn.Embedding(cfg.max_photos + 1, cfg.hidden)  # last slot = text token
        nn.init.normal_(self.pos.weight, std=0.02)
        if text_dim:
            self.text_proj = nn.Sequential(
                nn.LayerNorm(text_dim), nn.Linear(text_dim, cfg.hidden), nn.GELU(), nn.Dropout(cfg.dropout)
            )
        self.att_v = nn.Linear(cfg.hidden, 128)
        self.att_u = nn.Linear(cfg.hidden, 128)
        self.att_w = nn.Linear(128, 1)
        self.out = nn.Sequential(nn.LayerNorm(cfg.hidden + 1), nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden + 1, 1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.constant_(self.out[-1].bias, target_mean)

    def forward(self, bags, mask, text=None, return_attention: bool = False):
        h = self.proj(bags.float()) + self.pos.weight[: bags.shape[1]]
        if text is not None:
            t = self.text_proj(text.float()) + self.pos.weight[-1]
            h = torch.cat([h, t.unsqueeze(1)], dim=1)
            mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
        scores = self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))).squeeze(-1)
        att = scores.masked_fill(~mask, -1e4).softmax(-1)
        pooled = (att.unsqueeze(-1) * h).sum(1)
        n_photos = mask[:, : bags.shape[1]].sum(1, keepdim=True).float().log1p()
        pred = self.out(torch.cat([pooled, n_photos], dim=1)).squeeze(-1)
        return (pred, att) if return_attention else pred


def _predict(model, bags, mask, text, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(bags), batch_size):
            t = text[i : i + batch_size] if text is not None else None
            out.append(model(bags[i : i + batch_size], mask[i : i + batch_size], t).float().cpu().numpy())
    return np.concatenate(out)


def train_head(
    bags: torch.Tensor, mask: torch.Tensor, y: torch.Tensor, cfg: HeadConfig,
    text: torch.Tensor | None = None,
) -> GatedAttentionHead:
    """Train one head for ``cfg.epochs`` epochs (one-cycle LR) on tensors already on the device."""
    model = GatedAttentionHead(
        bags.shape[-1], cfg, float(y.mean()), text.shape[-1] if text is not None else 0
    ).to(bags.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * int(np.ceil(len(bags) / cfg.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps, pct_start=0.1)
    for _ in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(bags), device=bags.device)
        for i in range(0, len(perm), cfg.batch_size):
            b = perm[i : i + cfg.batch_size]
            m = mask[b]
            if cfg.photo_dropout > 0:
                keep = torch.rand(m.shape, device=m.device) > cfg.photo_dropout
                keep[:, 0] |= ~(m & keep).any(1)  # never drop every photo of a pet
                m = m & keep
            loss = nn.functional.mse_loss(model(bags[b], m, text[b] if text is not None else None), y[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    return model


def run_head_cv(
    bags_tr: torch.Tensor, mask_tr: torch.Tensor, bags_te: torch.Tensor, mask_te: torch.Tensor,
    y: np.ndarray, folds: np.ndarray, cfg: HeadConfig,
    text_tr: torch.Tensor | None = None, text_te: torch.Tensor | None = None,
    device: str = "cuda", verbose: bool = True,
) -> dict:
    """Train the head on each fold for a fixed number of epochs (cosine LR, no early stopping,
    so the validation fold never influences the model) → OOF, fold-averaged test preds, QWK."""
    use_text = cfg.use_text and text_tr is not None
    bags_tr, mask_tr = bags_tr.to(device), mask_tr.to(device)
    bags_te, mask_te = bags_te.to(device), mask_te.to(device)
    if use_text:
        text_tr, text_te = text_tr.to(device), text_te.to(device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)

    oof, pred_te = np.zeros(len(y)), np.zeros(len(bags_te))
    for f in np.unique(folds):
        seed_everything(cfg.seed + int(f))
        tr_idx = torch.from_numpy(np.where(folds != f)[0]).to(device)
        va_idx = np.where(folds == f)[0]
        model = train_head(bags_tr[tr_idx], mask_tr[tr_idx], y_t[tr_idx], cfg, text_tr[tr_idx] if use_text else None)
        oof[va_idx] = _predict(model, bags_tr[va_idx], mask_tr[va_idx], text_tr[va_idx] if use_text else None)
        pred_te += _predict(model, bags_te, mask_te, text_te if use_text else None) / len(np.unique(folds))
        if verbose:
            rmse = np.sqrt(np.mean((oof[va_idx] - y[va_idx]) ** 2))
            print(f"fold {f}: val rmse {rmse:.4f}")

    rounder = OptimizedRounder().fit(oof, y)
    score = qwk(y, rounder.predict(oof))
    if verbose:
        print(f"OOF QWK: {score:.4f}")
    return {"oof": oof, "test": pred_te, "qwk": score}
