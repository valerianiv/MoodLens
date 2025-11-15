import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import pandas as pd
from src.config import PROCESSED_DIR, RESULTS_DIR
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tensorflow.keras.models import load_model
warnings.filterwarnings('ignore')

os.makedirs(RESULTS_DIR, exist_ok=True)

# Загрузка нормализованных данных
X_train = np.load(os.path.join(PROCESSED_DIR, 'landmarks_train_norm.npy'))
y_train = np.load(os.path.join(PROCESSED_DIR, 'labels_train.npy'))
X_test = np.load(os.path.join(PROCESSED_DIR, 'landmarks_test_norm.npy'))
y_test = np.load(os.path.join(PROCESSED_DIR, 'labels_test.npy'))

print(f"Размеры: train={X_train.shape}, test={X_test.shape}")

# Модели
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    "SVM (RBF)": SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        class_weight='balanced',
        random_state=42
    ),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )
}

results = []

for name, model in models.items():
    print(f"\nОбучение {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Основные метрики
    accuracy = f1_score(y_test, y_pred, average='micro')  # Это и есть Top-1 Accuracy
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Per-class F1
    report = classification_report(y_test, y_pred, output_dict=True)
    per_class_f1 = {cls: report[str(cls)]['f1-score'] for cls in range(len(report)-3)}  # без avg / total
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"{name}")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Macro F1: {macro_f1:.4f}")
    print(f"   Weighted F1: {weighted_f1:.4f}")
    print(f"   Per-class F1: {per_class_f1}")
    
    # Сохраняем результат
    results.append({
        'Model': name,
        'Accuracy': round(accuracy, 4),
        'Macro F1': round(macro_f1, 4),
        'Weighted F1': round(weighted_f1, 4),
        **{f'F1_Class_{i}': round(per_class_f1[i], 4) for i in range(len(per_class_f1))}
    })
    
    # Сохраняем confusion matrix как CSV
    df_cm = pd.DataFrame(cm, index=[f"True_{i}" for i in range(len(cm))],
                         columns=[f"Pred_{i}" for i in range(len(cm))])
    df_cm.to_csv(os.path.join(RESULTS_DIR, f'cm_{name.replace(" ", "_").lower()}.csv'))
    
    # Визуализация confusion matrix (опционально)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(len(cm)), yticklabels=range(len(cm)))
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'cm_{name.replace(" ", "_").lower()}.png'))
    plt.close()

# Итоговая таблица
df = pd.DataFrame(results)
df.to_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'), index=False)
print("\nВсе результаты сохранены в папку 'results/'")
