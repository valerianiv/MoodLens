import os

# Определение корневой директории проекта
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Пути к основным директориям проекта
DATA_DIR = os.path.join(BASE_DIR, 'data', 'RAF-DB')  
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Эмоции в порядке RAF-DB 
EMOTION_LABELS = [
    "surprise",    # 0
    "fear",        # 1
    "disgust",     # 2
    "happiness",   # 3
    "sadness",     # 4
    "anger",       # 5
    "neutral"      # 6
]

# Словарь для быстрого преобразования эмоций в индексы
EMOTION_TO_IDX = {label: idx for idx, label in enumerate(EMOTION_LABELS)}