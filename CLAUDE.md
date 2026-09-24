# neoversity-dl-cv-final — контекст для Claude Code

## Задача
Kaggle: https://www.kaggle.com/competitions/deep-learning-for-computer-vision-and-nlp-2026-10
Урізана версія PetFinder.my Adoption Prediction (2019). Передбачити `AdoptionSpeed`
за текстом (`Description`) і фото. **Табличних ознак немає** — тільки PetID, Description, target.
Метрика: quadratic weighted kappa (QWK). Оцінка: QWK >= 0.5 → 100 балів, інакше `QWK × 200`.
+10 балів (не вище 100) за код + детальний опис рішення на форумі змагання протягом 2 днів
після закриття. Закриття ~4 жовтня 2026.

## Дані (перевірено в 01_eda)
- `data/raw/{train,test,sample_submission}.csv`; зображення в `data/raw/images/{train,test}/<PetID>-<n>.jpg`
  (у підпапках — `build_image_index()` шукає рекурсивно, колонка `split`).
- train: 6 431 рядок (`PetID, Description, AdoptionSpeed`); test: 1 891 (`PetID, Description`).
- **Класи 1–4 (класу 0 немає)**: 18.6 / 27.6 / 20.6 / 33.2 %. Мітки не зсуваємо — QWK інваріантна
  до зсуву; `OptimizedRounder(labels=(1,2,3,4))` повертає одразу 1–4.
- `PetID` — 9-символьний hex-рядок; читати як `str` (є ID на кшталт `10e723583`). У test.csv
  4 ID втратили ведучий нуль — `load_data()` робить `zfill(9)`. Після цього фото є в усіх тварин.
- У `images/test/` є 8 тварин, яких немає в test.csv — ігноруються.
- `sample_submission.csv` — лише 4 рядки-приклади; сабмішн будувати по ID з `test.csv`
  (формат `PetID,AdoptionSpeed`, з заголовком, порядок рядків не важливий).
- Тексти переважно англійською (є малайська/китайська), медіана ~46 слів; 5 порожніх у train, 1 у test.
  Фото ~400×400, у середньому ~4.4 на тварину.
- **Майже-дублікати описів** (char TF-IDF cos ≥ 0.9): 12.6 % train у групах, std таргету всередині
  групи 0.27 проти 1.12 глобально; 208 тварин test мають двійника в train. Тому фолди —
  **звичайні стратифіковані** (відтворюють випадковий поділ train/test), а дублікати
  використовуються через OOF kNN-target ознаки (`src/features.knn_target_features`).
  Будь-які target-залежні ознаки — тільки OOF по колонці `fold`.

## Результати (OOF QWK, LightGBM на фолдах)
- 02 текстові ознаки (regex+meta): 0.246; + kNN-target по тексту: 0.319.
- 03 ознаки з фото (zero-shot + якість, без kNN): 0.547; + kNN по CLIP: 0.558
  (строга nested-перевірка kNN: 0.566 → витоку немає). Найсильніші: zero-shot вік
  (baby/adult/old, |ρ|≈0.31–0.35), порода pure/mixed (|ρ|≈0.28), img_knn_y (ρ=0.49).
- 04 бейзлайн: ручні ознаки 02+03 → 0.576; + SVD/PCA → 0.578; + OOF Ridge (CLIP img 0.587,
  TF-IDF 0.335, CLIP text 0.281) як ознаки → 0.611; 3 сіди → **0.6125** (nested-перевірка 0.612).
  Сабмішн `submissions/04_baseline_lgb_stack.csv`. LB: _(ще не відправлено)_.

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
- `src/features.py` — regex-ознаки тексту (тип, вік у місяцях, порода, здоров'я…), групи
  майже-дублікатів, OOF kNN-target ознаки (для TF-IDF і для CLIP-ембедінгів).
- `src/images.py` — CLIP ViT-L/14 (`datacomp_xl_s13b_b90k`): ембедінги + якість фото за один
  прохід, zero-shot ознаки, агрегація по тварині (first/mean/max).
- `src/models.py` — `run_lgb_cv()` (LightGBM на фолдах → OOF, test, QWK, importance), `ridge_oof()`.
- `src/text.py` (TODO) — Dataset/fine-tune трансформера, `run_text_cv(train, test, cfg, device)`.
- `src/pet_adoption/` — залишок scaffold, не використовується.

## План
1. `01_eda` → 2. `02_text_features` (regex, дублікати, kNN-target → `text_feats_*.parquet`) →
3. `03_image_features` (CLIP, zero-shot, якість → `img_feats_*.parquet`, `clip_*.npy`) →
4. `04_baseline` (TF-IDF+SVD + ознаки 02/03 → LGBM) → 5. `05_text_models` (DeBERTa/XLM-R) →
6. `06_fusion_final` (стекінг, пороги, retrain на всіх даних, сабмішн)
→ опис рішення на форумі змагання (+10 балів).

Примітка: порада з опису змагання «перенавчити на train + test» стосується всіх *розмічених*
даних — test.csv без міток, тож фінальна модель = retrain на всьому train (усі фолди).
Pseudo-labeling test — лише як окремий експеримент із перевіркою на CV.
