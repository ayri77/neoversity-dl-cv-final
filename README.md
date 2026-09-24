# neoversity-dl-cv-final

Фінальний проєкт курсу Deep Learning for Computer Vision and NLP —
прогнозування швидкості адопції тварин (PetFinder.my), Kaggle.

## Структура
- `notebooks/` — ноутбуки за кроками (EDA → baseline → текст → зображення → фінал)
- `src/` — спільний код (конфіг, дані, метрика, моделі)
- `data/raw/` — сирі дані змагання (не в git)
- `data/processed/` — кешовані фічі та ембедінги
- `submissions/` — файли для подання

## Запуск (uv)
```bash
uv sync                                   # створює .venv і ставить залежності
uv run python -m ipykernel install --user --name neoversity-dl --display-name "neoversity-dl (uv)"
uv run jupyter lab
```
