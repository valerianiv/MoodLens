import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from datetime import datetime
import threading
import json
import uuid

class EmotionRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MoodLens - Система распознавания эмоций")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2c3e50')
        
        # Загрузка модели и рекомендаций
        self.model = None
        self.recommendations = {}
        self.history = []  # Хранение истории анализа
        self.load_resources()
        
        # Переменные
        self.image_path = None
        self.original_image = None
        self.processed_image = None
        self.face_coordinates = []
        self.current_face_index = -1
        self.emotion_results = []  # Результаты текущего анализа
        self.emotion_labels = ['Злость', 'Отвращение', 'Страх', 'Радость', 'Нейтрально', 'Грусть', 'Удивление']
        
        # Настройка стилей
        self.setup_styles()
        
        # Создание интерфейса
        self.create_widgets()
        
    def setup_styles(self):
        """Настройка стилей приложения"""
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#2c3e50')
        self.style.configure('TLabel', background='#2c3e50', foreground='white', font=('Arial', 10))
        self.style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground='#3498db')
        self.style.configure('TButton', font=('Arial', 10), padding=10)
        # Убираем белый текст — оставляем стандартный (чёрный по умолчанию для ttk)
        # self.style.configure('Primary.TButton', background='#3498db', foreground='white')  # ❌ УДАЛЕНО
        # self.style.configure('Success.TButton', background='#2ecc71', foreground='white')  # ❌ УДАЛЕНО
        # self.style.configure('Warning.TButton', background='#e74c3c', foreground='white')  # ❌ УДАЛЕНО
        
    def load_resources(self):
        """Загрузка модели и рекомендаций в отдельном потоке"""
        def load_in_thread():
            try:
                # Загрузка модели
                if os.path.exists('models/vgg16_emotion_model.h5'):
                    self.model = load_model('models/vgg16_emotion_model.h5')
                    # Загрузка рекомендаций
                    if os.path.exists('recommendations.json'):
                        with open('recommendations.json', 'r', encoding='utf-8') as f:
                            self.recommendations = json.load(f)
                    else:
                        # Резервные рекомендации
                        self.recommendations = {
                            'Злость': 'Рекомендуется глубокое дыхание и физическая активность для снятия напряжения.',
                            'Отвращение': 'Попробуйте сменить обстановку или сосредоточиться на позитивных аспектах.',
                            'Страх': 'Практикуйте техники заземления и разбейте проблему на маленькие шаги.',
                            'Радость': 'Продолжайте в том же духе! Поделитесь своим настроением с близкими.',
                            'Нейтрально': 'Стабильное состояние. Рекомендуется разнообразить рутину для поддержания баланса.',
                            'Грусть': 'Общение с друзьями, прогулки на свежем воздухе и хобби могут помочь.',
                            'Удивление': 'Используйте этот момент для нового опыта и обучения.'
                        }
                    # Загрузка истории
                    if os.path.exists('history.json'):
                        with open('history.json', 'r', encoding='utf-8') as f:
                            self.history = json.load(f)
                    self.update_status("✅ Модель, рекомендации и история загружены успешно")
                else:
                    self.update_status("❌ Модель не найдена. Запустите train_vgg16.py сначала")
            except Exception as e:
                self.update_status(f"❌ Ошибка загрузки: {str(e)}")
        
        threading.Thread(target=load_in_thread, daemon=True).start()
        
    def create_widgets(self):
        """Создание интерфейса приложения"""
        # Главный контейнер
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Верхнее меню (как на NTECHLAB)
        menu_frame = tk.Frame(main_container, bg='#2c3e50', height=50)
        menu_frame.pack(fill=tk.X, side=tk.TOP)
        
        # Логотип слева
        logo_label = tk.Label(menu_frame, text="MoodLens", font=("Arial", 14, "bold"), 
                             fg="#3498db", bg='#2c3e50')
        logo_label.pack(side=tk.LEFT, padx=20, pady=5)
        
        # Меню справа
        menu_items = ["Продукты", "Применение", "Компания", "Медиа"]
        for item in menu_items:
            btn = tk.Button(menu_frame, text=item, font=("Arial", 10), bg='#2c3e50', fg='white',
                           relief='flat', activebackground='#34495e', padx=15)
            btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Центральная область
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Левая панель - загрузка изображения
        left_frame = ttk.Frame(content_frame, width=600)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Правая панель - результаты
        right_frame = ttk.Frame(content_frame, width=600)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # === ЛЕВАЯ ПАНЕЛЬ ===
        # Панель загрузки изображения
        upload_frame = ttk.LabelFrame(left_frame, text="Загрузка изображения", padding=15)
        upload_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Кнопка с ЧЁРНЫМ текстом (стандартный стиль ttk)
        self.upload_btn = ttk.Button(upload_frame, text="📁 Загрузить фото", 
                                    command=self.load_image)
        self.upload_btn.pack(pady=5)
        
        # Отображение изображения
        self.image_frame = ttk.LabelFrame(left_frame, text="Предпросмотр", padding=10)
        self.image_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.image_frame, bg='#34495e', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # === ПРАВАЯ ПАНЕЛЬ ===
        # Панель управления
        control_frame = ttk.LabelFrame(right_frame, text="Управление", padding=15)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        control_buttons = ttk.Frame(control_frame)
        control_buttons.pack(fill=tk.X)
        
        # Все кнопки с ЧЁРНЫМ текстом
        self.detect_btn = ttk.Button(control_buttons, text="🔍 Найти лица", 
                                    command=self.detect_faces, state='disabled')
        self.detect_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.analyze_btn = ttk.Button(control_buttons, text="🧠 Анализ всех лиц", 
                                     command=self.analyze_all_faces, state='disabled')
        self.analyze_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.report_btn = ttk.Button(control_buttons, text="📊 Создать отчет PDF", 
                                    command=self.generate_report, state='disabled')
        self.report_btn.pack(side=tk.LEFT)
        
        # Панель результатов — ТОЛЬКО ОДНА ВКЛАДКА
        results_frame = ttk.LabelFrame(right_frame, text="Результаты анализа", padding=15)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Простой текстовый виджет (без notebook)
        self.results_text = tk.Text(results_frame, wrap=tk.WORD, width=60, height=25,
                                   bg='#ecf0f1', fg='#2c3e50', font=('Arial', 10))
        scrollbar = ttk.Scrollbar(results_frame, command=self.results_text.yview)
        self.results_text.config(yscrollcommand=scrollbar.set)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Статус бар
        self.status_var = tk.StringVar(value="Готов к работе...")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def update_status(self, message):
        """Обновление статусной строки"""
        self.status_var.set(message)
        self.root.update_idletasks()
        
    def load_image(self):
        """Загрузка изображения"""
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            self.image_path = file_path
            self.face_coordinates = []
            self.current_face_index = -1
            self.emotion_results = []
            
            try:
                # Загрузка изображения
                self.original_image = cv2.imread(file_path)
                if self.original_image is None:
                    raise ValueError("Не удалось загрузить изображение")
                
                # Конвертация для отображения
                image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                self.processed_image = image_rgb.copy()
                
                # Отображение изображения
                self.display_image(image_rgb)
                
                self.update_status("✅ Изображение загружено успешно")
                self.detect_btn.config(state='normal')
                self.analyze_btn.config(state='disabled')
                self.report_btn.config(state='disabled')
                
                # Очистка результатов
                self.results_text.delete(1.0, tk.END)
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка загрузки изображения: {str(e)}")
                self.update_status("❌ Ошибка загрузки изображения")
    
    def display_image(self, image):
        """Отображение изображения на canvas"""
        # Масштабирование изображения под размер canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 500
            canvas_height = 400
        
        h, w = image.shape[:2]
        scale = min(canvas_width/w, canvas_height/h)
        new_w, new_h = int(w*scale), int(h*scale)
        
        # Resize изображения
        image_resized = cv2.resize(image, (new_w, new_h))
        
        # Конвертация для tkinter
        image_pil = Image.fromarray(image_resized)
        self.photo = ImageTk.PhotoImage(image_pil)
        
        # Очистка canvas и отображение
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.photo, anchor=tk.CENTER)
        
        # Сохранение параметров масштабирования
        self.scale_factor = scale
        self.canvas_offset_x = (canvas_width - new_w) // 2
        self.canvas_offset_y = (canvas_height - new_h) // 2
    
    def detect_faces(self):
        """Обнаружение лиц на изображении"""
        if self.original_image is None:
            return
            
        self.update_status("🔍 Обнаружение лиц...")
        
        try:
            # Используем Haar cascade для обнаружения лиц
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                messagebox.showinfo("Информация", "Лица не обнаружены")
                self.update_status("❌ Лица не обнаружены")
                return
            
            self.face_coordinates = faces.tolist()
            self.current_face_index = 0
            
            # Отрисовка прямоугольников вокруг лиц
            image_with_faces = self.processed_image.copy()
            for i, (x, y, w, h) in enumerate(faces):
                color = (0, 255, 0) if i == 0 else (255, 0, 0)
                cv2.rectangle(image_with_faces, (x, y), (x+w, y+h), color, 2)
                cv2.putText(image_with_faces, f'Face {i+1}', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            self.display_image(image_with_faces)
            self.update_status(f"✅ Обнаружено лиц: {len(faces)}")
            self.analyze_btn.config(state='normal')
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка обнаружения лиц: {str(e)}")
            self.update_status("❌ Ошибка обнаружения лиц")
    
    def on_canvas_click(self, event):
        """Обработка клика по canvas для выбора лица"""
        if not self.face_coordinates:
            return
            
        # Конвертация координат клика в координаты оригинального изображения
        x_click = (event.x - self.canvas_offset_x) / self.scale_factor
        y_click = (event.y - self.canvas_offset_y) / self.scale_factor
        
        # Поиск лица по координатам
        for i, (x, y, w, h) in enumerate(self.face_coordinates):
            if x <= x_click <= x + w and y <= y_click <= y + h:
                self.current_face_index = i
                self.highlight_selected_face()
                self.update_status(f"✅ Выбрано лицо {i+1}")
                break
    
    def highlight_selected_face(self):
        """Подсветка выбранного лица"""
        if self.current_face_index == -1:
            return
            
        image_with_highlight = self.processed_image.copy()
        
        for i, (x, y, w, h) in enumerate(self.face_coordinates):
            if i == self.current_face_index:
                # Подсветка выбранного лица
                cv2.rectangle(image_with_highlight, (x, y), (x+w, y+h), (0, 255, 255), 3)
                cv2.putText(image_with_highlight, f'Face {i+1}', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.rectangle(image_with_highlight, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(image_with_highlight, f'Face {i+1}', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        self.display_image(image_with_highlight)
    
    def analyze_all_faces(self):
        """Анализ эмоций для всех обнаруженных лиц"""
        if not self.face_coordinates or self.model is None:
            return
            
        self.update_status("🧠 Анализ всех лиц...")
        
        try:
            self.emotion_results = []  # Очищаем старые результаты
            
            for i, (x, y, w, h) in enumerate(self.face_coordinates):
                face_roi = self.original_image[y:y+h, x:x+w]
                
                # Предобработка для модели
                face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, (48, 48))
                face_normalized = face_resized.astype('float32') / 255.0
                face_input = np.expand_dims(face_normalized, axis=-1)  # Добавляем канал
                face_input = np.expand_dims(face_input, axis=0)  # Добавляем batch dimension
                
                # Предсказание
                predictions = self.model.predict(face_input, verbose=0)
                emotion_probs = predictions[0]
                
                # Сохранение результатов
                result = {
                    'face_index': i,
                    'coordinates': (x, y, w, h),
                    'emotions': {},
                    'dominant_emotion': '',
                    'dominant_prob': 0,
                    'analysis_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'image_path': self.image_path
                }
                
                # Обработка результатов
                for j, emotion in enumerate(self.emotion_labels):
                    prob = emotion_probs[j] * 100
                    result['emotions'][emotion] = prob
                    if prob > result['dominant_prob']:
                        result['dominant_prob'] = prob
                        result['dominant_emotion'] = emotion
                
                self.emotion_results.append(result)
            
            # Сохраняем в историю
            self.save_to_history()
            
            # Отображаем первый результат
            if self.emotion_results:
                self.current_face_index = 0
                self.display_results(self.emotion_results[0])
                self.report_btn.config(state='normal')
                self.update_status(f"✅ Анализ завершен: {len(self.emotion_results)} лиц")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка анализа эмоций: {str(e)}")
            self.update_status("❌ Ошибка анализа эмоций")
    
    def display_results(self, result):
        """Отображение результатов анализа"""
        # Очистка предыдущих результатов
        self.results_text.delete(1.0, tk.END)
        
        # Детали эмоций
        self.results_text.insert(tk.END, f"АНАЛИЗ ЛИЦА {result['face_index'] + 1}\n")
        self.results_text.insert(tk.END, "="*50 + "\n\n")
        self.results_text.insert(tk.END, f"Доминирующая эмоция: {result['dominant_emotion']}\n")
        self.results_text.insert(tk.END, f"Уверенность: {result['dominant_prob']:.1f}%\n\n")
        self.results_text.insert(tk.END, "РАСПРЕДЕЛЕНИЕ ЭМОЦИЙ:\n")
        
        for emotion, prob in sorted(result['emotions'].items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob / 5)
            self.results_text.insert(tk.END, f"{emotion:<12}: {prob:5.1f}% {bar}\n")
    
    def save_to_history(self):
        """Сохранение результатов анализа в историю"""
        # Добавляем новые результаты в историю
        for result in self.emotion_results:
            # Генерируем уникальный ID
            result['id'] = str(uuid.uuid4())
            # Добавляем в историю
            self.history.append(result)
        
        # Сохраняем в файл
        try:
            with open('history.json', 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
            self.update_status("✅ История сохранена")
        except Exception as e:
            self.update_status(f"❌ Ошибка сохранения истории: {str(e)}")
    
    def generate_report(self):
        """Генерация PDF отчета для всех лиц"""
        if not self.emotion_results:
            messagebox.showwarning("Предупреждение", "Нет данных для отчета")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Сохранить отчет PDF",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")]
        )
        
        if file_path:
            try:
                self.update_status("📊 Создание PDF отчета...")
                
                with PdfPages(file_path) as pdf:
                    # Обложка
                    fig, ax = plt.subplots(figsize=(10, 12))
                    ax.axis('off')
                    ax.text(0.5, 0.6, 'MoodLens - Отчет анализа эмоций', 
                           fontsize=24, fontweight='bold', ha='center', va='center')
                    ax.text(0.5, 0.4, f'Дата создания: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                           fontsize=14, ha='center', va='center')
                    ax.text(0.5, 0.2, f'Изображение: {os.path.basename(self.image_path) if self.image_path else "Unknown"}',
                           fontsize=12, ha='center', va='center')
                    pdf.savefig(fig)
                    plt.close()
                    
                    # Для каждого лица
                    for i, result in enumerate(self.emotion_results):
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
                        fig.suptitle(f'Анализ лица {result["face_index"] + 1}', fontsize=16, fontweight='bold')
                        
                        # График эмоций
                        emotions = list(result['emotions'].keys())
                        probabilities = list(result['emotions'].values())
                        
                        colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#95a5a6', '#3498db', '#9b59b6']
                        bars = ax1.bar(emotions, probabilities, color=colors, alpha=0.7)
                        ax1.set_title('Распределение эмоций')
                        ax1.set_ylabel('Вероятность (%)')
                        ax1.tick_params(axis='x', rotation=45)
                        
                        # Добавление значений на столбцы
                        for bar, prob in zip(bars, probabilities):
                            height = bar.get_height()
                            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                                    f'{prob:.1f}%', ha='center', va='bottom')
                        
                        # Текстовая информация
                        report_text = f"""
Анализ лица {result['face_index'] + 1}
Дата анализа: {result['analysis_time']}

Доминирующая эмоция: {result['dominant_emotion']}
Уверенность: {result['dominant_prob']:.1f}%

Детальное распределение эмоций:
"""
                        # Добавляем детали по всем эмоциям
                        for emotion, prob in sorted(result['emotions'].items(), key=lambda x: x[1], reverse=True):
                            report_text += f"{emotion}: {prob:.1f}%\n"
                        
                        ax2.axis('off')
                        ax2.text(0.1, 0.9, report_text, transform=ax2.transAxes, fontsize=12, 
                                verticalalignment='top', linespacing=1.5)
                        
                        plt.tight_layout(rect=[0, 0, 1, 0.95])
                        pdf.savefig(fig)
                        plt.close()
                
                self.update_status(f"✅ Отчет сохранен: {file_path}")
                messagebox.showinfo("Успех", f"PDF отчет успешно сохранен!\n{file_path}")
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка создания отчета: {str(e)}")
                self.update_status("❌ Ошибка создания отчета")

def main():
    # Создание папки reports если не существует
    os.makedirs('reports', exist_ok=True)
    
    # Запуск приложения
    root = tk.Tk()
    app = EmotionRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()