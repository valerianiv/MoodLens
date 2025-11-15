# MoodLens

Система распознавания эмоций по лицу на основе **нормализованных координат ключевых точек**.

## Особенности
- Поддержка **нескольких лиц** в кадре
- Сравнение моделей: **RF, SVM, MLP, EfficientNet**
- Только датасет **FER2013Plus**
- Сохранение истории анализа в `history.json`

## Запуск
1. Подготовка: `python src\extract_landmarks.py`
2. Нормализация: `python -m src.normalize_landmarks`
3. Обучение: `python src\evaluate_models.py`
4. Интерфейс: `python src\app.py`