import os
import cv2
import numpy as np
from tqdm import tqdm
from mediapipe.python.solutions import face_mesh
from .config import DATA_DIR, PROCESSED_DIR, EMOTION_TO_IDX, EMOTION_LABELS

# Возвращение координат 68 ключевых точек лица 
def extract_single_landmarks(image_path):
    try:
        # Чтение изображения с обработкой ошибок 
        with open(image_path, "rb") as f:
            bytes_data = bytearray(f.read())
        # Проверка на то, пустой ли файл
        if not bytes_data:
            return []
        
        # Декодирование байтов в изображение OpenCV
        nparr = np.frombuffer(bytes_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return []

    # Проверка успеха декодирования
    if image is None:
        return []

    # Конвертация форматов изображения для корректной обработки MediaPipe
    if len(image.shape) == 2:
        # Преобразование из черно-белого изображения в rgb
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        # Преобразование из изображения с альфа-каналов в rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # Получение размеров изображения
    h, w = image.shape[:2]
    # Увеличение маленьких изображений для лучшей детекции лиц
    scale_factor = 2 if h < 100 or w < 100 else 1
    enlarged = cv2.resize(image, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_CUBIC)

    rgb = cv2.cvtColor(enlarged, cv2.COLOR_BGR2RGB)

    # Инициализация MediaPipe Face Mesh
    with face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.2,
        min_tracking_confidence=0.2
    ) as fm:
        # Обработка изображения и получение результатов
        results = fm.process(rgb)
        if not results.multi_face_landmarks:
            return []

        coords = []
        for i in range(68):
            # Берется первое найденное лицо
            lm = results.multi_face_landmarks[0].landmark[i]
            coords.extend([lm.x, lm.y])
        return [np.array(coords, dtype=np.float32)]


# Основная функция для обработки датасета RAF-DB
def process_rafdb():
    # Нахождение ключевых точек лица и сохранение в numpy файлы для обучения
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    dataset_dir = os.path.join(DATA_DIR, 'DATASET')
    # Маппинг номеров папок датасета
    label_map = {
        '1': 'surprise',  # Удивление
        '2': 'fear',      # Страх
        '3': 'disgust',   # Отвращение
        '4': 'happiness', # Радость
        '5': 'sadness',   # Грусть
        '6': 'anger',     # Злость
        '7': 'neutral'    # Безразличие
    }

    # Списки для train и test
    train_landmarks, train_labels = [], []
    test_landmarks, test_labels = [], []

    # Обработка обеих частей датасета, train и test 
    for split in ['train', 'test']:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            print(f"Папка {split} не найдена: {split_dir}")
            continue

        print(f"Обработка {split}")

        # Проход по всем папкам с эмоциями
        for label_folder in sorted(os.listdir(split_dir)):
            if label_folder not in label_map:
                continue

            # Получение текстового названия эмоции по номеру папки
            emotion_label = label_map[label_folder]
            label_idx = EMOTION_TO_IDX.get(emotion_label)
            if label_idx is None:
                continue

            # Путь к папке с изображениями конкретной эмоции
            folder_path = os.path.join(split_dir, label_folder)
            for img_name in tqdm(os.listdir(folder_path), desc=f"{split}/{label_folder}"):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                img_path = os.path.join(folder_path, img_name)
                faces = extract_single_landmarks(img_path)
                if not faces:
                    continue

                # Добавляем в нужный список
                for lm in faces:
                    if len(lm) == 136:
                        if split == 'train':
                            train_landmarks.append(lm)
                            train_labels.append(label_idx)
                        else:
                            test_landmarks.append(lm)
                            test_labels.append(label_idx)

    # Обучающая выборка
    np.save(os.path.join(PROCESSED_DIR, 'landmarks_train.npy'), np.array(train_landmarks))
    np.save(os.path.join(PROCESSED_DIR, 'labels_train.npy'), np.array(train_labels))
    # Тестовая выборка
    np.save(os.path.join(PROCESSED_DIR, 'landmarks_test.npy'), np.array(test_landmarks))
    np.save(os.path.join(PROCESSED_DIR, 'labels_test.npy'), np.array(test_labels))

    print(f"Train: {len(train_landmarks)} образцов")
    print(f"Test:  {len(test_landmarks)} образцов")


if __name__ == "__main__":
    process_rafdb()