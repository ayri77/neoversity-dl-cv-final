"""Fine-tuning a transformer encoder for AdoptionSpeed regression on fixed folds.

Design choices (common in text-regression Kaggle solutions, e.g. CommonLit):
  * mean pooling over tokens + a linear head, target centred on its mean;
  * backbone dropout disabled — dropout adds noise to regression outputs;
  * AdamW with a larger learning rate for the head, cosine schedule with warmup;
  * bf16 autocast, dynamic padding, evaluation several times per epoch and
    keeping the predictions (val + test) of the best checkpoint by val RMSE.
"""
from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from src.config import TARGET_COL, TEXT_COL
from src.utils import OptimizedRounder, qwk, seed_everything


@dataclass
class TextConfig:
    model_name: str = "microsoft/deberta-v3-base"
    max_len: int = 256
    batch_size: int = 16
    epochs: int = 3
    lr: float = 2e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    evals_per_epoch: int = 2
    grad_checkpointing: bool = False
    seed: int = 42


class TextDataset(Dataset):
    def __init__(self, encodings: dict, targets: np.ndarray | None = None):
        self.enc = encodings
        self.targets = targets

    def __len__(self) -> int:
        return len(self.enc["input_ids"])

    def __getitem__(self, i: int) -> dict:
        item = {k: v[i] for k, v in self.enc.items()}
        if self.targets is not None:
            item["labels"] = float(self.targets[i])
        return item


def make_collate(pad_id: int):
    """Dynamic padding to the longest sequence in the batch."""

    def collate(batch: list[dict]) -> dict:
        max_len = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for j, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[j, :n] = torch.tensor(b["input_ids"])
            mask[j, :n] = 1
        out = {"input_ids": ids, "attention_mask": mask}
        if "labels" in batch[0]:
            out["labels"] = torch.tensor([b["labels"] for b in batch], dtype=torch.float32)
        return out

    return collate


class TextRegressor(nn.Module):
    """Transformer encoder + masked mean pooling + linear regression head."""

    def __init__(self, model_name: str, target_mean: float = 0.0, grad_checkpointing: bool = False):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        for attr in ("hidden_dropout_prob", "attention_probs_dropout_prob", "dropout", "attention_dropout"):
            if hasattr(config, attr):
                setattr(config, attr, 0.0)
        # fp32 master weights: transformers 5 otherwise keeps the checkpoint dtype
        # (fp16 for deberta-v3), and AdamW on fp16 weights diverges to NaN
        self.backbone = AutoModel.from_pretrained(model_name, config=config, dtype=torch.float32)
        if grad_checkpointing:
            self.backbone.gradient_checkpointing_enable()
        self.head = nn.Linear(config.hidden_size, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, target_mean)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.head(pooled.float()).squeeze(-1)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    preds = []
    for b in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            p = model(b["input_ids"].to(device), b["attention_mask"].to(device))
        preds.append(p.float().cpu().numpy())
    return np.concatenate(preds)


def _sorted_loader(enc: dict, collate, batch_size: int) -> tuple[DataLoader, np.ndarray]:
    """Inference loader over length-sorted samples (less padding); returns inverse order."""
    order = np.argsort([len(x) for x in enc["input_ids"]])
    sub = {k: [v[i] for i in order] for k, v in enc.items()}
    return DataLoader(TextDataset(sub), batch_size=batch_size * 2, collate_fn=collate), np.argsort(order)


def train_fold(
    cfg: TextConfig, enc_tr: dict, y_tr: np.ndarray, enc_va: dict, y_va: np.ndarray,
    enc_te: dict, tokenizer, device: str,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Train on one fold → (best val preds, test preds at the same checkpoint, history)."""
    collate = make_collate(tokenizer.pad_token_id)
    train_loader = DataLoader(
        TextDataset(enc_tr, y_tr), batch_size=cfg.batch_size, shuffle=True, collate_fn=collate, drop_last=True
    )
    va_loader, va_inv = _sorted_loader(enc_va, collate, cfg.batch_size)
    te_loader, te_inv = _sorted_loader(enc_te, collate, cfg.batch_size)

    model = TextRegressor(cfg.model_name, float(y_tr.mean()), cfg.grad_checkpointing).to(device)
    no_decay = ("bias", "LayerNorm.weight", "layernorm", "norm")
    backbone_params = list(model.backbone.named_parameters())
    groups = [
        {"params": [p for n, p in backbone_params if not any(k in n for k in no_decay)],
         "lr": cfg.lr, "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in backbone_params if any(k in n for k in no_decay)],
         "lr": cfg.lr, "weight_decay": 0.0},
        {"params": model.head.parameters(), "lr": cfg.head_lr, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups)
    total_steps = cfg.epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(cfg.warmup_ratio * total_steps), total_steps)
    eval_steps = {
        int(round(len(train_loader) * (e + (k + 1) / cfg.evals_per_epoch)))
        for e in range(cfg.epochs) for k in range(cfg.evals_per_epoch)
    }

    best = {"rmse": math.inf}
    history, step, t0 = [], 0, time.time()
    loss_fn = nn.MSELoss()
    for epoch in range(cfg.epochs):
        model.train()
        for b in train_loader:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                pred = model(b["input_ids"].to(device), b["attention_mask"].to(device))
            loss = loss_fn(pred.float(), b["labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step in eval_steps:
                p_va = predict(model, va_loader, device)[va_inv]
                rmse = float(np.sqrt(np.mean((p_va - y_va) ** 2)))
                history.append({"step": step, "epoch": step / len(train_loader), "val_rmse": rmse,
                                 "train_loss": loss.item(), "min": (time.time() - t0) / 60})
                if rmse < best["rmse"]:
                    best = {"rmse": rmse, "va": p_va, "te": predict(model, te_loader, device)[te_inv]}
                model.train()

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return best["va"], best["te"], history


def run_text_cv(
    train: pd.DataFrame, test: pd.DataFrame, cfg: dict | TextConfig, device: str = "cuda",
    folds_to_run: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Fine-tune on each fold of ``train['fold']`` → (OOF, fold-averaged test preds, history).

    ``folds_to_run`` allows a quick check on a subset of folds; the OOF of
    skipped folds is left as NaN.
    """
    cfg = cfg if isinstance(cfg, TextConfig) else TextConfig(**cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    def encode(texts: pd.Series) -> dict:
        enc = tokenizer(texts.fillna("").tolist(), truncation=True, max_length=cfg.max_len)
        return {"input_ids": enc["input_ids"]}

    enc_all_tr, enc_te = encode(train[TEXT_COL]), encode(test[TEXT_COL])
    y = train[TARGET_COL].to_numpy(dtype=float)
    folds = train["fold"].to_numpy()
    fold_ids = folds_to_run if folds_to_run is not None else sorted(np.unique(folds))

    oof = np.full(len(train), np.nan)
    pred_te = np.zeros(len(test))
    history = []
    for f in fold_ids:
        seed_everything(cfg.seed + int(f))
        tr_idx, va_idx = np.where(folds != f)[0], np.where(folds == f)[0]
        pick = lambda enc, idx: {k: [v[i] for i in idx] for k, v in enc.items()}  # noqa: E731
        p_va, p_te, hist = train_fold(
            cfg, pick(enc_all_tr, tr_idx), y[tr_idx], pick(enc_all_tr, va_idx), y[va_idx],
            enc_te, tokenizer, device,
        )
        oof[va_idx] = p_va
        pred_te += p_te / len(fold_ids)
        history += [{"fold": f, **h} for h in hist]
        score = qwk(y[va_idx], OptimizedRounder().fit(p_va, y[va_idx]).predict(p_va))
        print(f"fold {f}: best val rmse {min(h['val_rmse'] for h in hist):.4f}  "
              f"fold QWK {score:.4f}  ({hist[-1]['min']:.1f} min)")
    return oof, pred_te, pd.DataFrame(history)
