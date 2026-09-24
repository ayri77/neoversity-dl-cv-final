# neoversity-dl-cv-final — контекст для Claude Code

## Задача
Kaggle: https://www.kaggle.com/competitions/deep-learning-for-computer-vision-and-nlp-2026-10
Урізана версія PetFinder.my Adoption Prediction (2019). Передбачити `AdoptionSpeed`
за текстом (`Description`) і фото. **Табличних ознак немає** — тільки PetID, Description, target.
Метрика: quadratic weighted kappa (QWK). 100 балів за QWK >= 0.5. Дедлайн ~4 жовтня 2026.
Дані: 6 432 об'єкти train; `data/raw/{train,test,sample_submission}.csv`, `data/raw/images/<PetID>-<n>.jpg`.

## Правила
- Оригінальні дані/мітки змагання 2019 **не використовувати** (це витік). Ідеї з публічних
  рішень — можна, з посиланням на джерело у фінальному описі.
- Робота ведеться людиною; Claude допомагає, пояснює, пише модулі та ячейки за запитом.

## Конвенції
- Рішення в ноутбуках `notebooks/0X_*.ipynb`; функції та класи виносити у `src/`
  (ноутбуки компактні, у них тільки оркестрація та результати).
- Markdown у ноутбуках — **українською**. Коментарі в коді та docstrings — **англійською**.
- Середовище: `uv` (`uv sync`, `uv run ...`), ядро Jupyter `neoversity-dl`.
- GPU: NVIDIA 12 GB → batch-size та max_len підбирати під це; використовувати AMP (bf16/fp16).
- Фолди фіксовані в `data/processed/train_meta.parquet` (колонка `fold`, 5 фолдів) — усі
  моделі валідуються на них. OOF/test-передбачення зберігати як
  `data/processed/oof_<name>.npy` / `test_<name>.npy` для стекінгу.
- Регресія + `OptimizedRounder` (src/utils.py) замість прямої класифікації.
- Сабмішни через `write_submission()` → `submissions/`.

## Модулі
- `src/config.py` — шляхи, константи. `src/utils.py` — seed, qwk, OptimizedRounder, write_submission.
- `src/data.py` — завантаження, індекс зображень, фолди, мета-ознаки тексту.
- `src/text.py` (TODO) — Dataset/fine-tune трансформера, `run_text_cv(train, test, cfg, device)`.
- `src/images.py` (TODO) — CLIP-ембедінги, агрегація по тварині, zero-shot ознаки.

## План
1. `01_eda` → 2. `02_baseline` (TF-IDF+SVD+meta → LGBM) → 3. `03_text_models` (DeBERTa/XLM-R) →
4. `04_image_models` (CLIP) → 5. `05_fusion_final` (стекінг, пороги, retrain на всіх даних, сабмішн)
→ опис рішення на форумі змагання (+10 балів).
