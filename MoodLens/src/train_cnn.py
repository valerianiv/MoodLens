import os
import numpy as np
import pandas as pd
import cv2
import random
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Dropout, Flatten, Dense, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import EfficientNetB0
import warnings
warnings.filterwarnings('ignore')

from src.config import DATA_DIR, RESULTS_DIR

# Определение базовых директорий проекта
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Создание необходимых директорий, если они не существуют
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Установление seed=42 для воспроизводимости результата
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

set_seed(42)  

# Загрузка данных из датасета RAF-DB
def load_rafdb_data(label_csv, image_dir, img_size=(48, 48)):
    # Проверка существования csv-файла
    if not os.path.exists(label_csv):
        raise FileNotFoundError(f"CSV файл не найден: {label_csv}")
    df = pd.read_csv(label_csv, header=None, names=['filename', 'label'])
    X, y = [], []
    # Прохождение по всем строкам csv
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Загрузка изображений"):
        filename = row['filename']
        img_path = os.path.join(image_dir, filename)
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # Изменение размера изображения до стандартного (48х48 пикселей)
        img = cv2.resize(img, img_size)
        X.append(img)
        y.append(int(row['label']) - 1)
        
    # Проверка, что данные были загружены
    if len(X) == 0:
        raise ValueError("Ни одно изображение не было загружено.")
    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y)
    return X, y

# Создание архитектуры simplecnn из 4 сверточных слоев с возрастающим количеством фильтров
def build_simple_cnn(input_shape=(48, 48, 1), num_classes=7):
    return Sequential([
        # Блок 1. Первичная обработка признаков
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2),
        Dropout(0.25),
        # Блок 2. Извлечение более сложных признаков
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2),
        Dropout(0.25),
        # Блок 3. Извлечение высокоуровневых признаков
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2),
        Dropout(0.25),
        # Блок 4. Абстрактные признаки
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2),
        Dropout(0.25),
        # Преобразование признаков в вероятности классов
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

# Создание архитектуры ferconvnet 
def build_ferconvnet(input_shape=(48, 48, 1), num_classes=7):
    return Sequential([
        # Первый сверточный блок
        Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2),
        Dropout(0.25),
        
        # Второй сверточный блок
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2),
        Dropout(0.25),
        
        # Третий сверточный блок
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2),
        Dropout(0.25),
        
        # Полносвязные слои для классификатора
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

# Создание модели на основе efficientnetb0
def build_efficientnet_b0(input_shape=(48, 48, 1), num_classes=7):
    base = EfficientNetB0(weights=None, include_top=False, input_shape=input_shape)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base.input, outputs=output)

# Определение путей к данным
train_csv = os.path.join(DATA_DIR, 'train_labels.csv')
test_csv = os.path.join(DATA_DIR, 'test_labels.csv')
train_img_dir = os.path.join(DATA_DIR, 'DATASET', 'train')
test_img_dir = os.path.join(DATA_DIR, 'DATASET', 'test')

# Загрузка данных
print("Загрузка обучающих данных")
X_train, y_train = load_rafdb_data(train_csv, train_img_dir)
print("Загрузка тестовых данных")
X_test, y_test = load_rafdb_data(test_csv, test_img_dir)

X_train = X_train.reshape(-1, 48, 48, 1)
X_test = X_test.reshape(-1, 48, 48, 1)
# Определение количества уникальных классов
num_classes = len(np.unique(y_train))

# Преобразование меток в ohe-hot encoding формат
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Балансировка
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# Обучение моделей 
models_cnn = {
    "SimpleCNN": build_simple_cnn(num_classes=num_classes),
    "FerConvNet": build_ferconvnet(num_classes=num_classes),
    "EfficientNetB0_48": build_efficientnet_b0(num_classes=num_classes)
}

results_cnn = []

# Обучение cnn моделей
for name, model in models_cnn.items():
    print(f"\nОбучение модели: {name}")
    # Определение функции потерь и оптимизатора
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Обучение модели
    history = model.fit(
        X_train, y_train_cat,
        batch_size=64,
        epochs=50,
        validation_data=(X_test, y_test_cat),
        class_weight=class_weight_dict,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=1
    )

    # Прогнозирование на тестовых данных
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    # Расчет метрик
    accuracy = f1_score(y_test, y_pred, average='micro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Отчет по метрикам
    report = classification_report(y_test, y_pred, output_dict=True)
    per_class_f1 = {cls: report[str(cls)]['f1-score'] for cls in range(num_classes)}
    
    # Матрица ошибок для анализа ошибок классификации
    cm = confusion_matrix(y_test, y_pred)

    print(f"{name} — Macro F1: {macro_f1:.4f}")

    # Сохранение модели
    model_path = os.path.join(MODELS_DIR, f'{name.lower().replace(" ", "_")}.keras')
    base_name = name.lower().replace(" ", "_")
    model.save(os.path.join(MODELS_DIR, f'{base_name}.keras'))  
    model.save(os.path.join(MODELS_DIR, f'{base_name}.h5'))    
    print(f"Модель сохранена в двух форматах: {base_name}.keras и {base_name}.h5")
    
    # Сохранение результатов
    results_cnn.append({
        'Model': name,
        'Accuracy': round(accuracy, 4),
        'Macro F1': round(macro_f1, 4),
        'Weighted F1': round(weighted_f1, 4),
        **{f'F1_Class_{i}': round(per_class_f1[i], 4) for i in range(num_classes)}
    })

    # Confusion matrix
    cm_path = os.path.join(RESULTS_DIR, f'cm_{name.replace(" ", "_").lower()}.csv')
    pd.DataFrame(cm).to_csv(cm_path, index=False)

    # Визуализация и сохранение матрицы ошибок как изображение
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.title(f'Матрица ошибок - {name}')
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.tight_layout()
    # Сохранение графика
    plt.savefig(os.path.join(RESULTS_DIR, f'cm_{name.replace(" ", "_").lower()}.png'))
    plt.close()

# Итоговая таблица
df_cnn = pd.DataFrame(results_cnn)
df_cnn.to_csv(os.path.join(RESULTS_DIR, 'cnn_comparison.csv'), index=False)
print(f"\nВсе модели обучены и сохранены в: {MODELS_DIR}")
print(f"Результаты сравнения: {os.path.join(RESULTS_DIR, 'cnn_comparison.csv')}")