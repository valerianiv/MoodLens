import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

RESULTS_DIR = 'results'

model_files = {
    "SimpleCNN": "cm_simplecnn.csv",
    "FerConvNet": "cm_ferconvnet.csv",
    "EfficientNetB0_48": "cm_efficientnetb0_48.csv",
    "Random Forest": "cm_random_forest.csv",
    "SVM (RBF)": "cm_svm_(rbf).csv",
    "MLP": "cm_mlp.csv"
}

class_names = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
results = []

for model_name, cm_file in model_files.items():
    cm_path = os.path.join(RESULTS_DIR, cm_file)
    if not os.path.exists(cm_path):
        print(f"Пропущена модель: {cm_path}")
        continue

    # Определяем формат по первой строке файла
    with open(cm_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()

    if first_line.startswith('True_0') or first_line.startswith(',Pred_0'):
        # Формат с заголовками (как у SVM, MLP, RF с True_0)
        df = pd.read_csv(cm_path, index_col=0)
        cm = df.values
    else:
        # Формат без заголовков (первая строка = "0,1,2,...")
        df = pd.read_csv(cm_path, header=0)  # первая строка — заголовки столбцов
        cm = df.values  # теперь 7x7

    if cm.shape != (7, 7):
        print(f"Матрица {model_name} имеет размер {cm.shape}, пропускаем.")
        continue

    # Восстанавливаем y_true и y_pred
    y_true, y_pred = [], []
    for i in range(7):
        for j in range(7):
            count = int(cm[i, j])
            y_true.extend([i] * count)
            y_pred.extend([j] * count)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    macro_f1 = f1_score(y_true, y_pred, average='macro')
    accuracy = f1_score(y_true, y_pred, average='micro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    f1_per_class = f1_score(y_true, y_pred, average=None)

    results.append({
        'Model': model_name,
        'Accuracy': round(accuracy, 4),
        'Macro F1': round(macro_f1, 4),
        'Weighted F1': round(weighted_f1, 4),
        **{f'F1_{class_names[i]}': round(f1_per_class[i], 4) for i in range(7)}
    })

# Вывод
df = pd.DataFrame(results).sort_values('Macro F1', ascending=False)
print("\nСравнение по macro F1")
print("=" * 110)
print(df.to_string(index=False, float_format="%.4f"))

df.to_csv(os.path.join(RESULTS_DIR, 'final_comparison_by_macro_f1.csv'), index=False)

# График
plt.figure(figsize=(12, 6))
bars = plt.bar(df['Model'], df['Macro F1'], color='teal', edgecolor='black')
plt.title('Сравнение моделей по Macro F1', fontsize=14)
plt.ylabel('Macro F1')
plt.xticks(rotation=45, ha='right')
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{bar.get_height():.3f}', ha='center')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'macro_f1_comparison.png'))
plt.show()

# Heatmap лучшей модели
best_model = df.iloc[0]['Model']
cm_path = os.path.join(RESULTS_DIR, model_files[best_model])

with open(cm_path, 'r', encoding='utf-8') as f:
    first_line = f.readline().strip()

if first_line.startswith('True_0') or first_line.startswith(',Pred_0'):
    cm_best = pd.read_csv(cm_path, index_col=0).values
else:
    cm_best = pd.read_csv(cm_path, header=0).values

plt.figure(figsize=(8, 7))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title(f'Confusion Matrix — {best_model}')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, f'cm_best_{best_model.replace(" ", "_").lower()}.png'))
plt.show()

print(f"\nЛучшая модель: {best_model} (Macro F1 = {df.iloc[0]['Macro F1']:.4f})")