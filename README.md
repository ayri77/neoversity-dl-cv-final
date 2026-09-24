# neoversity-dl-cv-final

Фінальний проєкт курсу Deep Learning for Computer Vision and NLP (Neoversity):
Kaggle-змагання [Deep Learning for CV and NLP 2026-10](https://www.kaggle.com/competitions/deep-learning-for-computer-vision-and-nlp-2026-10) —
прогнозування швидкості адопції тварин PetFinder.my (`AdoptionSpeed`, класи 1–4)
**лише за текстовим описом і фотографіями**. Метрика — quadratic weighted kappa (QWK).

## Результат

| | OOF QWK (5 фолдів) | Public LB |
|---|---|---|
| Бейзлайн (`04_baseline`) | 0.6125 | 0.7912 |
| **Фінал** (`final_submission`) | **0.6210** (чесна оцінка 0.6124) | **0.8088** |

Коротко про рішення: табличних ознак немає, тому вік, тип і породу тварини **відновлюємо
з фото** (zero-shot CLIP / SigLIP2) і тексту (регулярні вирази). Майже-дублікати оголошень
використовуємо через out-of-fold kNN-target. Далі — моделі 1-го рівня (Ridge на ембедінгах,
DeBERTa-v3, attention-голова на ембедінгах усіх фото) і стекінг LightGBM з підбором порогів.
Детальний опис: [docs/forum_post.md](docs/forum_post.md).

## Структура

```
notebooks/
  final_submission.ipynb   ← фінальний конвеєр: сирі дані → submissions/submission.csv
  01_eda.ipynb             розвідувальний аналіз
  02_text_features.ipynb   regex-ознаки, майже-дублікати, kNN-target по тексту
  03_image_features.ipynb  CLIP та SigLIP2: zero-shot, якість фото, kNN-target
  04_baseline.ipynb        бейзлайн: ознаки + Ridge-стекінг → LightGBM, абляція
  05_text_models.ipynb     fine-tuning DeBERTa-v3-base
  06_image_head.ipynb      attention-голова на ембедінгах усіх фото
  07_fusion_final.ipynb    фінальний стекінг, пороги, перенавчання, хід експериментів
src/
  config.py  data.py  utils.py      шляхи, завантаження, QWK, OptimizedRounder, сабмішн
  features.py                       ознаки тексту, майже-дублікати, OOF kNN-target
  images.py                         CLIP / SigLIP2: ембедінги, zero-shot, агрегація
  text.py  image_head.py  models.py DeBERTa, attention-голова, LightGBM / Ridge
  pipeline.py                       етапи фінального конвеєра з зафіксованими параметрами
scripts/run_nb.py                   запуск ноутбука з live-логом у logs/
tests/                              швидкі тести (uv run pytest)
docs/forum_post.md                  опис рішення для форуму змагання
```

Ноутбуки `01`–`07` — дослідницька частина (аналіз, абляції, вибір параметрів);
кожен закінчується розділом «Висновки».

## Відтворення

1. Середовище ([uv](https://docs.astral.sh/uv/), NVIDIA GPU 12 GB, CUDA-збірка PyTorch ставиться автоматично):
   ```bash
   uv sync
   uv run python -m ipykernel install --user --name neoversity-dl --display-name "neoversity-dl (uv)"
   ```
2. Дані змагання розкласти так:
   ```
   data/raw/train.csv  data/raw/test.csv  data/raw/sample_submission.csv
   data/raw/images/train/<PetID>-<n>.jpg  data/raw/images/test/<PetID>-<n>.jpg
   ```
3. Запустити фінальний ноутбук (~45 хв з нуля; повторно — кілька хвилин завдяки кешу `data/final_cache/`):
   ```bash
   uv run python scripts/run_nb.py notebooks/final_submission.ipynb
   ```
   Результат — `submissions/submission.csv`. Прогрес: `logs/final_submission.log`.

Ваги моделей (CLIP ViT-L/14 DataComp, SigLIP2 SO400M, DeBERTa-v3-base) завантажуються
з Hugging Face при першому запуску.

## Правила

Оригінальні дані й мітки PetFinder 2019 не використовувались. Джерела ідей і моделей
перелічені в [docs/forum_post.md](docs/forum_post.md).
