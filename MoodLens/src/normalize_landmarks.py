import os
import numpy as np
from src.config import PROCESSED_DIR

# Нормализует landmarks: центрирует по носу и масштабирует по межзрачковому расстоянию
def normalize_landmarks(X):
 
    # Список для хранения нормализированных векторов
    normalized = []
    for lm in X:
        coords = lm.reshape(68, 2)  # (68, 2)
        # Центрирование относительно носа
        nose = coords[30]
        centered = coords - nose

        # Масштаб - расстояние между глазами (36 - левый угол, 45 - правый)
        left_eye = coords[36]
        right_eye = coords[45]
        # Вычисление евклидова расстояния между глазами
        eye_dist = np.linalg.norm(left_eye - right_eye)
        # Порог 1e-6 предотвращает деление на 0
        if eye_dist > 1e-6:
            centered = centered / eye_dist
        # Создание одномерного массива из всех координат
        normalized.append(centered.flatten())
    return np.array(normalized)

# Загрузка
X_train = np.load(os.path.join(PROCESSED_DIR, 'landmarks_train.npy'))
X_test = np.load(os.path.join(PROCESSED_DIR, 'landmarks_test.npy'))

# Нормализация
X_train_norm = normalize_landmarks(X_train)
X_test_norm = normalize_landmarks(X_test)

# Сохранение
np.save(os.path.join(PROCESSED_DIR, 'landmarks_train_norm.npy'), X_train_norm)
np.save(os.path.join(PROCESSED_DIR, 'landmarks_test_norm.npy'), X_test_norm)
print("Нормализация завершена")