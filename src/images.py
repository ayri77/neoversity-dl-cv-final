"""CLIP image embeddings, simple photo-quality stats and zero-shot features.

One pass over all photos gives, per image:
  * an L2-normalised CLIP embedding (later aggregated per pet),
  * cheap quality statistics (size, brightness, contrast, sharpness, colourfulness).
Zero-shot features are then computed from the cached embeddings with text
prompts, so prompts can be changed without re-reading any image.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.config import ID_COL

CLIP_MODEL = "ViT-L-14"
CLIP_PRETRAINED = "datacomp_xl_s13b_b90k"

QUALITY_COLS = ["img_w", "img_h", "img_aspect", "img_kb", "brightness", "contrast", "sharpness", "colorfulness"]


def image_quality(img: Image.Image, file_bytes: int) -> np.ndarray:
    """Cheap photo-quality statistics on a downscaled copy of the image."""
    w, h = img.size
    small = img.copy()
    small.thumbnail((256, 256))
    gray = small.convert("L")
    g = np.asarray(gray, dtype=np.float32)
    edges = np.asarray(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    rgb = np.asarray(small, dtype=np.float32)
    # Hasler & Süsstrunk colourfulness metric
    rg = rgb[..., 0] - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    colorful = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return np.array(
        [w, h, w / h, file_bytes / 1024, g.mean(), g.std(), edges.var(), colorful], dtype=np.float32
    )


class PetImageDataset(Dataset):
    """Returns (CLIP-preprocessed tensor, quality stats) for each image path."""

    def __init__(self, paths: list[str], preprocess):
        self.paths = list(paths)
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        path = self.paths[i]
        with open(path, "rb") as f:
            n_bytes = len(f.read())
        img = Image.open(path).convert("RGB")
        return self.preprocess(img), torch.from_numpy(image_quality(img, n_bytes))


def load_clip(model_name: str = CLIP_MODEL, pretrained: str = CLIP_PRETRAINED, device: str = "cuda"):
    """Load an open_clip model in eval mode → (model, preprocess, tokenizer)."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    return model, preprocess, open_clip.get_tokenizer(model_name)


@torch.no_grad()
def extract_image_features(
    paths: list[str],
    model,
    preprocess,
    device: str = "cuda",
    batch_size: int = 128,
    num_workers: int = 4,
) -> tuple[np.ndarray, pd.DataFrame]:
    """One pass over images → (normalised CLIP embeddings float16, quality DataFrame)."""
    loader = DataLoader(
        PetImageDataset(paths, preprocess),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )
    embs, quals = [], []
    for x, q in tqdm(loader, desc="images"):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            e = model.encode_image(x.to(device, non_blocking=True))
        embs.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy().astype(np.float16))
        quals.append(q.numpy())
    return np.concatenate(embs), pd.DataFrame(np.concatenate(quals), columns=QUALITY_COLS)


@torch.no_grad()
def encode_texts(texts: list[str], model, tokenizer, device: str = "cuda", batch_size: int = 256) -> np.ndarray:
    """Normalised CLIP text embeddings (texts are truncated to 77 tokens)."""
    out = []
    for i in range(0, len(texts), batch_size):
        tok = tokenizer(texts[i : i + batch_size]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            e = model.encode_text(tok)
        out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


# --------------------------------------------------------------------------- #
# Zero-shot features
# --------------------------------------------------------------------------- #
# group → {label: [prompts]}; probabilities are a softmax over labels within a group
ZERO_SHOT_PROMPTS: dict[str, dict[str, list[str]]] = {
    "type": {
        "cat": ["a photo of a cat", "a photo of a kitten"],
        "dog": ["a photo of a dog", "a photo of a puppy"],
    },
    "age": {
        "baby": ["a photo of a tiny newborn kitten", "a photo of a tiny newborn puppy"],
        "young": ["a photo of a young kitten", "a photo of a young puppy"],
        "adult": ["a photo of an adult cat", "a photo of an adult dog"],
        "old": ["a photo of an old senior cat", "a photo of an old senior dog"],
    },
    "breed": {
        "pure": ["a photo of a purebred pedigree pet", "a photo of a Persian cat",
                 "a photo of a Siamese cat", "a photo of a husky", "a photo of a poodle",
                 "a photo of a shih tzu", "a photo of a golden retriever"],
        "mixed": ["a photo of a mixed breed pet", "a photo of a stray street cat",
                  "a photo of a mongrel street dog", "a photo of a domestic short hair cat"],
    },
    "fur": {
        "short": ["a photo of a short-haired pet"],
        "long": ["a photo of a long-haired fluffy pet"],
    },
    "color": {
        "black": ["a photo of a black pet"], "white": ["a photo of a white pet"],
        "orange": ["a photo of an orange ginger pet"], "brown": ["a photo of a brown pet"],
        "grey": ["a photo of a grey pet"], "calico": ["a photo of a calico or tabby pet"],
    },
    "count": {
        "one": ["a photo of a single animal"],
        "many": ["a photo of several animals together", "a photo of a litter of kittens",
                 "a photo of a litter of puppies"],
    },
    "health": {
        "healthy": ["a photo of a healthy pet"],
        "sick": ["a photo of a sick or injured pet", "a photo of a pet with a skin disease",
                 "a photo of a pet with a wound"],
    },
    "scene": {
        "home": ["a photo of a pet at home"],
        "cage": ["a photo of a pet in a cage"],
        "street": ["a photo of a pet on the street"],
        "vet": ["a photo of a pet at a veterinary clinic"],
    },
    "human": {
        "no": ["a photo of an animal alone"],
        "yes": ["a photo of a person holding an animal"],
    },
    "quality": {
        "good": ["a professional high quality photo of a pet", "a cute photo of a pet"],
        "bad": ["a blurry low quality photo", "a dark photo", "a screenshot of text", "a poster with text"],
    },
}


def zero_shot_features(
    img_emb: np.ndarray, model, tokenizer, device: str = "cuda",
    prompts: dict[str, dict[str, list[str]]] | None = None, logit_scale: float = 100.0,
) -> pd.DataFrame:
    """Per-image zero-shot probabilities, columns ``zs_<group>_<label>``."""
    prompts = prompts or ZERO_SHOT_PROMPTS
    emb = torch.from_numpy(img_emb.astype(np.float32))
    cols = {}
    for group, labels in prompts.items():
        # Average the prompt embeddings of each label (prompt ensembling)
        class_emb = []
        for plist in labels.values():
            t = torch.from_numpy(encode_texts(plist, model, tokenizer, device)).mean(0)
            class_emb.append(torch.nn.functional.normalize(t, dim=-1))
        probs = (logit_scale * emb @ torch.stack(class_emb).T).softmax(-1).numpy()
        for j, label in enumerate(labels):
            cols[f"zs_{group}_{label}"] = probs[:, j]
    return pd.DataFrame(cols)


# --------------------------------------------------------------------------- #
# Per-pet aggregation
# --------------------------------------------------------------------------- #
def aggregate_per_pet(
    image_index: pd.DataFrame, feats: pd.DataFrame, aggs: tuple[str, ...] = ("first", "mean", "max")
) -> pd.DataFrame:
    """Aggregate per-image features to pet level.

    ``image_index`` must be sorted by (PetID, img_idx) and aligned with ``feats``;
    "first" is the profile photo (img_idx == 1), which buyers see first.
    """
    df = pd.concat([image_index[[ID_COL]].reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    out = df.groupby(ID_COL, sort=False).agg(list(aggs))
    out.columns = [f"{c}_{a}" for c, a in out.columns]
    return out.reset_index()


def pet_embeddings(image_index: pd.DataFrame, img_emb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-pet (PetIDs, first-photo embedding, re-normalised mean embedding)."""
    ids = image_index[ID_COL].to_numpy()
    uniq, start = np.unique(ids, return_index=True)  # index is sorted by PetID, img_idx
    emb = img_emb.astype(np.float32)
    mean = np.add.reduceat(emb, start, axis=0)
    mean /= np.linalg.norm(mean, axis=1, keepdims=True)
    return uniq, emb[start], mean


# --------------------------------------------------------------------------- #
# Full pipeline for an additional encoder
# --------------------------------------------------------------------------- #
def encoder_feature_pipeline(
    tag: str,
    model_name: str,
    pretrained: str,
    image_index: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    out_dir,
    device: str = "cuda",
    batch_size: int = 64,
    num_workers: int = 6,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Embeddings → zero-shot → per-pet aggregation → OOF kNN-target → text-image similarity.

    Everything is prefixed with ``tag`` so features of several encoders can be
    combined. Photo-quality stats are not repeated (they are encoder-independent
    and already come from the CLIP pass). Saves to ``out_dir``:
    ``{tag}_img_emb.npy`` (cache), ``{tag}_pet_{mean,first}_{split}.npy``,
    ``{tag}_text_{split}.npy`` and ``img_feats_{tag}_{split}.parquet``.
    """
    from src.config import TARGET_COL, TEXT_COL
    from src.features import knn_target_features

    emb_path = out_dir / f"{tag}_img_emb.npy"
    model, preprocess, tokenizer = load_clip(model_name, pretrained, device)
    if use_cache and emb_path.exists():
        img_emb = np.load(emb_path)
    else:
        img_emb, _ = extract_image_features(
            image_index["path"].tolist(), model, preprocess, device, batch_size, num_workers
        )
        np.save(emb_path, img_emb)
    assert len(img_emb) == len(image_index)

    # Zero-shot with the model's own temperature (SigLIP ≈ 110, CLIP ≈ 100)
    zs = zero_shot_features(img_emb, model, tokenizer, device, logit_scale=float(model.logit_scale.exp()))
    pet_zs = aggregate_per_pet(image_index, zs, aggs=("first", "mean", "max"))

    pet_ids, emb_first, emb_mean = pet_embeddings(image_index, img_emb)
    pos = pd.Series(np.arange(len(pet_ids)), index=pet_ids)
    tr_pos, te_pos = pos[train[ID_COL]].to_numpy(), pos[test[ID_COL]].to_numpy()
    for split, p in (("train", tr_pos), ("test", te_pos)):
        np.save(out_dir / f"{tag}_pet_mean_{split}.npy", emb_mean[p])
        np.save(out_dir / f"{tag}_pet_first_{split}.npy", emb_first[p])

    y = train[TARGET_COL].to_numpy(dtype=float)
    knn_tr, knn_te = knn_target_features(emb_mean[tr_pos], emb_mean[te_pos], y, train["fold"].to_numpy(), prefix="img")

    feats = {}
    for split, df, p, knn in (("train", train, tr_pos, knn_tr), ("test", test, te_pos, knn_te)):
        t = encode_texts(df[TEXT_COL].str.slice(0, 300).tolist(), model, tokenizer, device)
        np.save(out_dir / f"{tag}_text_{split}.npy", t)
        out = df[[ID_COL]].merge(pet_zs, on=ID_COL, how="left")
        out[knn.columns] = knn.to_numpy()
        out["txt_img_mean"] = (t * emb_mean[p]).sum(1)
        out["txt_img_first"] = (t * emb_first[p]).sum(1)
        out = out.rename(columns={c: f"{tag}_{c}" for c in out.columns if c != ID_COL})
        out.to_parquet(out_dir / f"img_feats_{tag}_{split}.parquet", index=False)
        feats[split] = out

    del model
    torch.cuda.empty_cache()
    return feats["train"], feats["test"]


# --------------------------------------------------------------------------- #
# Self-supervised encoder (no text tower): DINOv2
# --------------------------------------------------------------------------- #
DINO_MODEL = "facebook/dinov2-large"


def dino_preprocess():
    """DINOv2 eval transform: resize 256 (bicubic) → center crop 224 → ImageNet normalisation."""
    from torchvision import transforms as T

    return T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


@torch.no_grad()
def extract_dino_embeddings(
    paths: list[str], model_name: str = DINO_MODEL, device: str = "cuda",
    batch_size: int = 128, num_workers: int = 6,
) -> np.ndarray:
    """Per-image DINOv2 features: [CLS ‖ mean of patch tokens], each part L2-normalised, float16."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_name, dtype=torch.float32).to(device).eval()
    loader = DataLoader(PetImageDataset(paths, dino_preprocess()), batch_size=batch_size,
                        num_workers=num_workers, pin_memory=True)
    out = []
    for x, _ in tqdm(loader, desc="DINOv2"):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            h = model(pixel_values=x.to(device, non_blocking=True)).last_hidden_state.float()
        cls = torch.nn.functional.normalize(h[:, 0], dim=-1)
        patches = torch.nn.functional.normalize(h[:, 1:].mean(1), dim=-1)
        out.append(torch.cat([cls, patches], dim=1).cpu().numpy().astype(np.float16))
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out)
