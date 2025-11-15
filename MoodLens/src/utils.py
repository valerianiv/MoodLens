import json
import os
from datetime import datetime

def save_history_entry(results_list, image_path, output_path='processed/history.json'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    entry = {
        'timestamp': datetime.now().isoformat(),
        'image_path': image_path,
        'faces': results_list  # список: [{'face_index': 0, 'emotion': 'happy', 'confidence': 0.92}, ...]
    }
    history = []
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
    history.append(entry)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)