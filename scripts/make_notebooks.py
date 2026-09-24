"""Generate notebook templates (run once; then edit notebooks directly)."""
from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NB_DIR.mkdir(exist_ok=True)

# Common header cell: make `src` importable regardless of cwd
SETUP = """import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import *
from src.utils import seed_everything, qwk, OptimizedRounder, write_submission
from src.data import load_data, build_image_index, add_photo_count, add_folds, text_meta_features

seed_everything(SEED)
pd.set_option("display.max_colwidth", 200)
sns.set_theme(style="whitegrid")"""


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def save(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, NB_DIR / name)
    print("wrote", name)


# ----------------------------------------------------------------------------
# 01 — EDA
# ----------------------------------------------------------------------------
save("01_eda.ipynb", [
    md("""
# 01 · Розвідувальний аналіз даних (EDA)

**Задача:** передбачити швидкість адопції тварини (`AdoptionSpeed`) на PetFinder.my
за текстовим описом та фотографіями. Метрика — *quadratic weighted kappa* (QWK).

**Мета цього ноутбука:** зрозуміти структуру даних, розподіл цільової змінної,
особливості текстів і зображень, щоб обґрунтовано спланувати моделювання.
"""),
    code(SETUP),
    md("## 1. Завантаження даних"),
    code("""
train, test = load_data()
print("train:", train.shape, " test:", test.shape)
print("test має таргет:", TARGET_COL in test.columns)
train.head(3)"""),
    md("## 2. Цільова змінна\n\nПеревіряємо діапазон класів (0–4 чи 1–4) та дисбаланс."),
    code("""
vc = train[TARGET_COL].value_counts().sort_index()
display(vc.to_frame("count").assign(share=lambda d: (d["count"] / len(train)).round(3)))
vc.plot.bar(title="Розподіл AdoptionSpeed");"""),
    md("## 3. Текстові описи\n\nДовжина, порожні описи, мова, приклади по класах."),
    code("""
meta = text_meta_features(train[TEXT_COL])
train_m = pd.concat([train, meta], axis=1)
display(meta.describe().T)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(train_m["desc_words"].clip(upper=300), bins=50, ax=axes[0]).set_title("Кількість слів")
sns.boxplot(data=train_m, x=TARGET_COL, y="desc_words", ax=axes[1]).set(ylim=(0, 300), title="Слова vs клас")
plt.tight_layout();"""),
    code("""
# Empty descriptions and non-ASCII share by target class
train_m.groupby(TARGET_COL)[["desc_is_empty", "desc_non_ascii", "desc_words"]].mean().round(3)"""),
    code("""
# A few random examples per class to get a feel for the texts
for c in sorted(train[TARGET_COL].unique()):
    print(f"=== AdoptionSpeed = {c} ===")
    for t in train.loc[train[TARGET_COL] == c, TEXT_COL].sample(2, random_state=SEED):
        print(" -", t[:250].replace("\\n", " "))
    print()"""),
    md("## 4. Зображення\n\nСкільки фото на тварину, скільки тварин без фото, розміри зображень."),
    code("""
img_index = build_image_index()
print("images:", len(img_index), " pets with photos:", img_index[ID_COL].nunique())
img_index.head()"""),
    code("""
train_m = add_photo_count(train_m, img_index)
test_p = add_photo_count(test, img_index)

print("train без фото:", (train_m["n_photos"] == 0).mean().round(3))
print("test  без фото:", (test_p["n_photos"] == 0).mean().round(3))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(x=train_m["n_photos"].clip(upper=10), ax=axes[0]).set_title("Фото на тварину")
train_m.groupby("n_photos")[TARGET_COL].mean().loc[:10].plot(ax=axes[1], marker="o", title="Середній клас vs n_photos")
plt.tight_layout();"""),
    code("""
# Image sizes on a random sample
from PIL import Image

sizes = [Image.open(p).size for p in img_index["path"].sample(200, random_state=SEED)]
pd.DataFrame(sizes, columns=["w", "h"]).describe().T"""),
    code("""
# Show several photos with their descriptions
sample = train_m[train_m["n_photos"] > 0].sample(6, random_state=SEED)
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, (_, row) in zip(axes.ravel(), sample.iterrows()):
    path = img_index.loc[img_index[ID_COL] == row[ID_COL], "path"].iloc[0]
    ax.imshow(Image.open(path)); ax.axis("off")
    ax.set_title(f"speed={row[TARGET_COL]}  n_photos={row['n_photos']}\\n{row[TEXT_COL][:60]}...", fontsize=9)
plt.tight_layout();"""),
    md("## 5. Збереження фолдів та мета-ознак\n\nФолди фіксуємо один раз, щоб усі наступні ноутбуки використовували однакову валідацію."),
    code("""
train_f = add_folds(train_m)
train_f.to_parquet(PROCESSED_DIR / "train_meta.parquet", index=False)
test_p.to_parquet(PROCESSED_DIR / "test_meta.parquet", index=False)
img_index.to_parquet(PROCESSED_DIR / "image_index.parquet", index=False)
train_f["fold"].value_counts().sort_index()"""),
    md("""
## Висновки

_(заповнити після аналізу)_

- Розподіл таргету: …
- Тексти: …
- Зображення: …
- Що це означає для моделювання: …
"""),
])

# ----------------------------------------------------------------------------
# 02 — Baseline
# ----------------------------------------------------------------------------
save("02_baseline.ipynb", [
    md("""
# 02 · Бейзлайн: TF-IDF + мета-ознаки → регресія → оптимізація порогів

**Ідея:** QWK — порядкова метрика, тому передбачаємо неперервне значення
і підбираємо пороги (`OptimizedRounder`) на OOF-передбаченнях. Це стабільно
краще за пряму класифікацію.

Ознаки: TF-IDF (слова + символи) → усічення SVD, статистики тексту, кількість фото.
Модель: LightGBM (регресія).
"""),
    code(SETUP),
    code("""
import lightgbm as lgb
from scipy.sparse import hstack, csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

train = pd.read_parquet(PROCESSED_DIR / "train_meta.parquet")
test = pd.read_parquet(PROCESSED_DIR / "test_meta.parquet")
print(train.shape, test.shape)"""),
    md("## 1. Текстові ознаки: TF-IDF → SVD"),
    code("""
N_SVD = 120
all_text = pd.concat([train[TEXT_COL], test[TEXT_COL]])

tfidf_word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50_000, sublinear_tf=True)
tfidf_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=5, max_features=80_000, sublinear_tf=True)

X_text = hstack([tfidf_word.fit_transform(all_text), tfidf_char.fit_transform(all_text)]).tocsr()
svd = TruncatedSVD(n_components=N_SVD, random_state=SEED)
X_svd = svd.fit_transform(X_text)
print("explained var:", svd.explained_variance_ratio_.sum().round(3))

svd_cols = [f"svd_{i}" for i in range(N_SVD)]
train_svd = pd.DataFrame(X_svd[: len(train)], columns=svd_cols)
test_svd = pd.DataFrame(X_svd[len(train):], columns=svd_cols)"""),
    md("## 2. Матриця ознак"),
    code("""
meta_cols = ["n_photos", "desc_len", "desc_words", "desc_upper_ratio", "desc_excl",
             "desc_digits", "desc_is_empty", "desc_non_ascii"]

X_train = pd.concat([train[meta_cols].reset_index(drop=True), train_svd], axis=1)
X_test = pd.concat([test[meta_cols].reset_index(drop=True), test_svd], axis=1)
y = train[TARGET_COL].values
X_train.shape, X_test.shape"""),
    md("## 3. Крос-валідація LightGBM (регресія)"),
    code("""
params = dict(
    objective="regression", learning_rate=0.02, num_leaves=31,
    feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=1.0, min_data_in_leaf=20, verbose=-1, seed=SEED,
)

oof = np.zeros(len(train))
pred_test = np.zeros(len(test))

for f in range(N_FOLDS):
    tr_idx, va_idx = np.where(train["fold"] != f)[0], np.where(train["fold"] == f)[0]
    dtr = lgb.Dataset(X_train.iloc[tr_idx], y[tr_idx])
    dva = lgb.Dataset(X_train.iloc[va_idx], y[va_idx])
    model = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
    oof[va_idx] = model.predict(X_train.iloc[va_idx])
    pred_test += model.predict(X_test) / N_FOLDS
    print(f"fold {f}: best_iter={model.best_iteration}  rmse={np.sqrt(np.mean((oof[va_idx]-y[va_idx])**2)):.4f}")"""),
    md("## 4. Оптимізація порогів та QWK"),
    code("""
n_classes = train[TARGET_COL].nunique()
rounder = OptimizedRounder(n_classes=n_classes).fit(oof, y)
print("thresholds:", rounder.coef_.round(3))
print("OOF QWK (optimised):", round(qwk(y, rounder.predict(oof)), 4))
print("OOF QWK (naive round):", round(qwk(y, np.clip(np.round(oof), y.min(), y.max())), 4))"""),
    code("""
# Save OOF / test predictions for stacking later
np.save(PROCESSED_DIR / "oof_baseline.npy", oof)
np.save(PROCESSED_DIR / "test_baseline.npy", pred_test)"""),
    md("## 5. Сабмішн"),
    code("""
write_submission(test[ID_COL], rounder.predict(pred_test), "01_baseline_tfidf_lgb");"""),
    md("## Висновки\n\n_(OOF QWK, LB score, що працює / не працює, наступні кроки)_"),
])

# ----------------------------------------------------------------------------
# 03 — Text models
# ----------------------------------------------------------------------------
save("03_text_models.ipynb", [
    md("""
# 03 · Текстові моделі: fine-tuning трансформера

Fine-tune попередньо навченого трансформера на регресію `AdoptionSpeed`
за 5 фолдами. OOF-передбачення зберігаємо як ознаку для фінального стекінгу.

Кандидати: `microsoft/deberta-v3-small`, `xlm-roberta-base` (є малайська/китайська),
`distilbert-base-uncased` (швидкий бейзлайн).
"""),
    code(SETUP),
    code("""
import torch
from src.text import TextRegressor, run_text_cv   # to be implemented in src/text.py

train = pd.read_parquet(PROCESSED_DIR / "train_meta.parquet")
test = pd.read_parquet(PROCESSED_DIR / "test_meta.parquet")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)"""),
    md("## 1. Конфігурація експерименту"),
    code("""
cfg = dict(
    model_name="microsoft/deberta-v3-small",
    max_len=256,
    batch_size=32,
    epochs=3,
    lr=3e-5,
)
cfg"""),
    md("## 2. Крос-валідація"),
    code("""
# oof, pred_test = run_text_cv(train, test, cfg, device=device)
# rounder = OptimizedRounder(n_classes=train[TARGET_COL].nunique()).fit(oof, train[TARGET_COL].values)
# print("OOF QWK:", qwk(train[TARGET_COL], rounder.predict(oof)))"""),
    md("## 3. Збереження OOF для стекінгу"),
    code("""
# tag = cfg["model_name"].split("/")[-1]
# np.save(PROCESSED_DIR / f"oof_text_{tag}.npy", oof)
# np.save(PROCESSED_DIR / f"test_text_{tag}.npy", pred_test)"""),
    md("## Висновки\n\n_(порівняння моделей, OOF QWK, час навчання)_"),
])

# ----------------------------------------------------------------------------
# 04 — Image models
# ----------------------------------------------------------------------------
save("04_image_models.ipynb", [
    md("""
# 04 · Зображення: CLIP-ембедінги та zero-shot ознаки

**План:**
1. Витягнути CLIP-ембедінги для всіх фото, агрегувати по тварині (mean / max / перше фото).
2. Zero-shot ознаки: кіт/собака, кошеня/доросла, одна/кілька тварин тощо.
3. Схожість опису та фото у спільному просторі CLIP.
4. (опційно) Fine-tune CNN/ViT-голови на регресію.

Ембедінги кешуємо у `data/processed/`, щоб рахувати один раз.
"""),
    code(SETUP),
    code("""
import torch
from src.images import extract_clip_embeddings, aggregate_per_pet, zero_shot_features  # to be implemented in src/images.py

img_index = pd.read_parquet(PROCESSED_DIR / "image_index.parquet")
train = pd.read_parquet(PROCESSED_DIR / "train_meta.parquet")
test = pd.read_parquet(PROCESSED_DIR / "test_meta.parquet")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device, len(img_index))"""),
    md("## 1. CLIP-ембедінги зображень"),
    code("""
# emb = extract_clip_embeddings(img_index["path"].tolist(), model_name="ViT-B-32", device=device)
# np.save(PROCESSED_DIR / "clip_img_emb.npy", emb)"""),
    md("## 2. Агрегація по тварині та zero-shot ознаки"),
    code("""
# pet_emb = aggregate_per_pet(img_index, emb)          # DataFrame: PetID + emb columns
# zs = zero_shot_features(emb, prompts={...})           # DataFrame: PetID + p_cat, p_kitten, ...
# pet_emb.to_parquet(PROCESSED_DIR / "clip_pet_features.parquet", index=False)"""),
    md("## 3. Швидка перевірка сигналу: LightGBM тільки на image-ознаках"),
    code("""
# TODO: reuse CV loop from 02 on image features only → OOF QWK"""),
    md("## Висновки\n\n_(чи є сигнал у зображеннях, які ознаки корисні)_"),
])

# ----------------------------------------------------------------------------
# 05 — Fusion & final
# ----------------------------------------------------------------------------
save("05_fusion_final.ipynb", [
    md("""
# 05 · Об'єднання модальностей, стекінг та фінальний сабмішн

Збираємо всі ознаки та OOF-передбачення з попередніх ноутбуків:
- мета-ознаки + TF-IDF/SVD (02)
- OOF трансформера (03)
- CLIP-ознаки зображень (04)

Модель другого рівня → оптимізація порогів → перенавчання на всіх даних → сабмішн.
"""),
    code(SETUP),
    md("## 1. Збір ознак"),
    code("""
train = pd.read_parquet(PROCESSED_DIR / "train_meta.parquet")
test = pd.read_parquet(PROCESSED_DIR / "test_meta.parquet")
y = train[TARGET_COL].values

stack_train, stack_test = {}, {}
for name in ["baseline"]:   # extend: "text_deberta-v3-small", "image_clip", ...
    stack_train[name] = np.load(PROCESSED_DIR / f"oof_{name}.npy")
    stack_test[name] = np.load(PROCESSED_DIR / f"test_{name}.npy")
stack_train = pd.DataFrame(stack_train); stack_test = pd.DataFrame(stack_test)
print(stack_train.corr().round(3))"""),
    md("## 2. Модель другого рівня"),
    code("""
# TODO: Ridge / LGBM on stacked OOF + selected raw features, CV, OptimizedRounder"""),
    md("## 3. Фінальний сабмішн"),
    code("""
# write_submission(test[ID_COL], final_preds, "final_stack");"""),
    md("""
## Підсумок проєкту

_(таблиця експериментів: модель → OOF QWK → LB; ключові висновки; що б робили далі)_
"""),
])
