import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

# Определение директории с результатами
RESULTS_DIR = 'results'

# Словарь соответствия названий модели и файлов с их матрицами ошибок
model_files = {
    "SimpleCNN": "cm_simplecnn.csv",
    "FerConvNet": "cm_ferconvnet.csv",
    "EfficientNetB0_48": "cm_efficientnetb0_48.csv",
    "Random Forest": "cm_random_forest.csv",
    "SVM (RBF)": "cm_svm_(rbf).csv",
    "MLP": "cm_mlp.csv"
}

# Названия классов эмоций в порядке, соответствующем 0-6
class_names = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
# Список для сохранения результатов всех моделей
results = []

# Проход по всем моделям для загрузки и анализа результатов
for model_name, cm_file in model_files.items():
    # Формирование полного пути к файлу матрицы ошибок
    cm_path = os.path.join(RESULTS_DIR, cm_file)
    if not os.path.exists(cm_path):
        print(f"Пропущена модель: {cm_path}")
        continue

    with open(cm_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()

    # Обработка двух возможных форматов матрицы ошибок
    if first_line.startswith('True_0') or first_line.startswith(',Pred_0'):
        # Формат csv с заголовками строк и столбцов
        df = pd.read_csv(cm_path, index_col=0)
        cm = df.values
    else:
        # Формат csv без заголовков 
        df = pd.read_csv(cm_path, header=0) 
        cm = df.values  # теперь 7x7

    # Проверка корректности размеров матрицы ошибок
    if cm.shape != (7, 7):
        print(f"Матрица {model_name} имеет размер {cm.shape}, пропускаем.")
        continue

    # Восстановление истинных и предсказанных меток из матрицы ошибок 
    y_true, y_pred = [], []
    
    for i in range(7):
        for j in range(7):
            count = int(cm[i, j])
            y_true.extend([i] * count)
            y_pred.extend([j] * count)

    # Преобразование списков в numpy массивы для совместимости и sklearn
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Расчет метрик
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    accuracy = f1_score(y_true, y_pred, average='micro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    f1_per_class = f1_score(y_true, y_pred, average=None)

    # Сохранение результатов модели
    results.append({
        'Model': model_name,
        'Accuracy': round(accuracy, 4),
        'Macro F1': round(macro_f1, 4),
        'Weighted F1': round(weighted_f1, 4),
        **{f'F1_{class_names[i]}': round(f1_per_class[i], 4) for i in range(7)}
    })

# преобразование результатов в dataframe и сортировка по убыванию macro-F1
df = pd.DataFrame(results).sort_values('Macro F1', ascending=False)
print("\nСравнение по macro F1")
print("=" * 110)
print(df.to_string(index=False, float_format="%.4f"))

df.to_csv(os.path.join(RESULTS_DIR, 'final_comparison_by_macro_f1.csv'), index=False)

# Столбчатая диаграмма сравнения моделей
plt.figure(figsize=(12, 6))
bars = plt.bar(df['Model'], df['Macro F1'], color='teal', edgecolor='black')
plt.title('Сравнение моделей по Macro F1', fontsize=14)
plt.ylabel('Macro F1')
plt.xticks(rotation=45, ha='right')
# Добавление значений на вершины столбцов
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{bar.get_height():.3f}', ha='center')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'macro_f1_comparison.png'))
plt.show()

# Тепловая карта матрицы ошибок лучшей модели
best_model = df.iloc[0]['Model']
cm_path = os.path.join(RESULTS_DIR, model_files[best_model])

# Определение формата файла
with open(cm_path, 'r', encoding='utf-8') as f:
    first_line = f.readline().strip()

if first_line.startswith('True_0') or first_line.startswith(',Pred_0'):
    cm_best = pd.read_csv(cm_path, index_col=0).values
else:
    cm_best = pd.read_csv(cm_path, header=0).values

# Создание тепловой карты матрицы ошибок
plt.figure(figsize=(8, 7))
# Создание тепловой карты с помощью seaborn 
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
# Настройка заголовка и подписей осей
plt.title(f'Матрица ошибок - {best_model}')
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, f'cm_best_{best_model.replace(" ", "_").lower()}.png'))
plt.show()

# Финальный вывод
print(f"\nЛучшая модель: {best_model} (Macro F1 = {df.iloc[0]['Macro F1']:.4f})")