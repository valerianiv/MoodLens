import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from datetime import datetime
import json
import uuid
from mediapipe.python.solutions import face_detection
class EmotionRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MoodLens - Система распознавания эмоций")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2c3e50')

        # Установка иконки
        try:
            icon_path = os.path.join(os.path.dirname(__file__), 'icon.png')
            if os.path.exists(icon_path):
                icon = ImageTk.PhotoImage(Image.open(icon_path).resize((32, 32)))
                self.root.iconphoto(True, icon)
        except Exception as e:
            print(f"Не удалось установить иконку: {e}")
        
        # Переменные
        self.image_path = None
        self.original_image = None
        self.processed_image = None
        self.face_bboxes = []
        self.current_face_index = -1
        self.emotion_results = []
        self.emotion_labels = ['Удивление', 'Страх', 'Отвращение', 'Радость', 'Грусть', 'Злость', 'Безразличие']
        self.history = []
        self.model = None
        self.load_resources()
        self.setup_styles()
        self.create_widgets()
        

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#2c3e50')
        self.style.configure('TLabel', background='#2c3e50', foreground='white', font=('Consolas', 10))
        self.style.configure('Title.TLabel', font=('Consolas', 16, 'bold'), foreground='#3498db')
        self.style.configure('TButton', font=('Consolas', 10), padding=10)

    def load_resources(self):
        try:
            model_path = os.path.join('models', 'simplecnn.h5') 
            if os.path.exists(model_path):
                self.model = load_model(model_path)
                if os.path.exists('history.json'):
                    with open('history.json', 'r', encoding='utf-8') as f:
                        self.history = json.load(f)
                print("123")
            else:
                print("345")
        except Exception as e:
            print("856")

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if not file_path:
            return

        self.image_path = file_path
        self.face_bboxes = []
        self.emotion_results = []
        self.current_face_index = -1

        # Блокируем кнопки, связанные с анализом
        self.detect_btn.config(state='disabled')
        self.analyze_btn.config(state='disabled')
        self.report_btn.config(state='disabled')

        # Очищаем поле результатов
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state='disabled')

        try:
            from PIL import Image
            pil_img = Image.open(file_path).convert('RGB')
            self.original_image = np.array(pil_img)[:, :, ::-1]  # RGB → BGR

            if self.original_image is None or self.original_image.size == 0:
                raise ValueError("Изображение пустое или повреждено")

            self.processed_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.display_image(self.processed_image)

            self.update_status("Изображение загружено")
            self.detect_btn.config(state='normal')  # Только кнопка "Найти лица" активна

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить изображение:\n{str(e)}")
            self.update_status("Ошибка загрузки изображения")

    def create_widgets(self):
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        title_frame = ttk.Frame(main_container)
        title_frame.pack(fill=tk.X, side=tk.TOP, pady=(0, 20))

        # Только текст "MoodLens" по центру
        title_label = tk.Label(title_frame, text="MoodLens", 
                            font=("Consolas", 24, "bold"), 
                            fg="#3498db", bg='#2c3e50')
        title_label.pack(expand=True)  # Центрируем по горизонтали

        # Основной контент с использованием grid для точного контроля
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Настройка grid для точного распределения - РАВНЫЕ ЧАСТИ
        content_frame.grid_rowconfigure(0, weight=0)  # Верхний ряд - компактный
        content_frame.grid_rowconfigure(1, weight=1)  # Нижний ряд - растягивается
        content_frame.grid_columnconfigure(0, weight=1, uniform="equal")  # Левая колонка - РАВНАЯ
        content_frame.grid_columnconfigure(1, weight=1, uniform="equal")  # Правая колонка - РАВНАЯ
        
        # Левая верхняя - Загрузка изображения
        upload_frame = ttk.LabelFrame(content_frame, text="Загрузка изображения", padding=15)
        upload_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        
        # Контейнер для кнопки загрузки 
        upload_buttons = ttk.Frame(upload_frame)
        upload_buttons.pack(fill=tk.BOTH, expand=True)
        
        # Используем grid для выравнивания как в управлении
        upload_buttons.grid_columnconfigure(0, weight=1)
        upload_buttons.grid_rowconfigure(0, weight=1)
        
        self.upload_btn = ttk.Button(upload_buttons, text="Загрузить фото", command=self.load_image)
        self.upload_btn.grid(row=0, column=0, sticky="nsew", padx=5, pady=10)

        # Правая верхняя - Управление (выровнено по верху с загрузкой)
        control_frame = ttk.LabelFrame(content_frame, text="Управление", padding=15)
        control_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=(0, 10))
        
        # Контейнер для равномерного распределения кнопок
        control_buttons = ttk.Frame(control_frame)
        control_buttons.pack(fill=tk.BOTH, expand=True)
        
        # Используем grid для равномерного распределения кнопок
        control_buttons.grid_columnconfigure(0, weight=1)
        control_buttons.grid_columnconfigure(1, weight=1)
        control_buttons.grid_columnconfigure(2, weight=1)
        control_buttons.grid_rowconfigure(0, weight=1)
        
        self.detect_btn = ttk.Button(control_buttons, text="Найти лица", command=self.detect_faces, state='disabled')
        self.detect_btn.grid(row=0, column=0, sticky="nsew", padx=5, pady=10)
        
        self.analyze_btn = ttk.Button(control_buttons, text="Анализ всех лиц", command=self.analyze_all_faces, state='disabled')
        self.analyze_btn.grid(row=0, column=1, sticky="nsew", padx=5, pady=10)
        
        self.report_btn = ttk.Button(control_buttons, text="Создать отчет PDF", command=self.generate_report, state='disabled')
        self.report_btn.grid(row=0, column=2, sticky="nsew", padx=5, pady=10)
        
        # Левая нижняя - Предпросмотр (большая высота)
        self.image_frame = ttk.LabelFrame(content_frame, text="Предпросмотр", padding=10)
        self.image_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10), pady=0)
        
        self.canvas = tk.Canvas(self.image_frame, bg='#34495e', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Правая нижняя - Результаты анализа (большая высота)
        results_frame = ttk.LabelFrame(content_frame, text="Результаты анализа", padding=10)
        results_frame.grid(row=1, column=1, sticky="nsew", padx=(10, 0), pady=0)
        
        self.results_text = tk.Text(results_frame, wrap=tk.WORD,
                                bg='#ecf0f1', fg='#2c3e50', font=('Consolas', 10), state='disabled')
        scrollbar = ttk.Scrollbar(results_frame, command=self.results_text.yview)
        self.results_text.config(yscrollcommand=scrollbar.set)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Статус бар
        self.status_var = tk.StringVar(value="Готов к работе...")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    def update_status(self, message):
            self.status_var.set(message)
        
    def detect_faces(self):
        if self.original_image is None:
            print("!")
            return
        if self.model is None:
            print("1")
            return
        self.update_status("🔍 Обнаружение лиц...")
        try:
            # Конвертация в RGB
            rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            self.face_bboxes = []

            # Используем FaceDetection вместо FaceMesh
            with face_detection.FaceDetection(
                model_selection=1,  # 0 — быстрее, 1 — точнее
                min_detection_confidence=0.3
            ) as fd:
                results = fd.process(rgb)

            if not results.detections:
                messagebox.showinfo("Информация", "Лица не обнаружены")
                self.update_status("Лица не обнаружены")
                return

            for detection in results.detections:
                # Получаем bounding box
                bbox = detection.location_data.relative_bounding_box
                x_min = int(bbox.xmin * w)
                y_min = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                self.face_bboxes.append((x_min, y_min, width, height))

            # Отображение
            img_with_faces = self.processed_image.copy()
            
            colors = [
                (0, 255, 0),    # Зелёный
                (255, 0, 0),    # Красный
                (0, 0, 255),    # Синий
                (255, 255, 0),  # Голубой
                (255, 0, 255),  # Пурпурный
                (0, 255, 255),  # Жёлтый
                (255, 165, 0),  # Оранжевый
            ]

            for i, (x, y, w_box, h_box) in enumerate(self.face_bboxes):
                color = colors[i % len(colors)]  # Циклически выбираем цвет
                cv2.rectangle(img_with_faces, (x, y), (x + w_box, y + h_box), color, 2)
                cv2.putText(img_with_faces, f'Face {i+1}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            self.display_image(img_with_faces)
            self.update_status(f"Обнаружено лиц: {len(self.face_bboxes)}")
            self.analyze_btn.config(state='normal')

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка обнаружения: {str(e)}")
            self.update_status("Ошибка обнаружения лиц")
    def on_canvas_click(self, event):
        if not self.face_bboxes:
            return

        x_click = (event.x - self.canvas_offset_x) / self.scale_factor
        y_click = (event.y - self.canvas_offset_y) / self.scale_factor

        for i, (x, y, w, h) in enumerate(self.face_bboxes):
            if x <= x_click <= x + w and y <= y_click <= y + h:
                self.current_face_index = i
                self.highlight_selected_face()
                self.update_status(f"Выбрано лицо {i+1}")
                break

    def highlight_selected_face(self):
        if self.current_face_index == -1:
            return

        img_highlight = self.processed_image.copy()
        for i, (x, y, w, h) in enumerate(self.face_bboxes):
            color = (0, 255, 255) if i == self.current_face_index else (255, 0, 0)
            thickness = 3 if i == self.current_face_index else 2
            cv2.rectangle(img_highlight, (x, y), (x + w, y + h), color, thickness)
            cv2.putText(img_highlight, f'Face {i+1}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5 if i != self.current_face_index else 0.6, color, thickness)

        self.display_image(img_highlight)

    def display_image(self, image):
        # Получаем размеры canvas
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w <= 0 or canvas_h <= 0:
            canvas_w = 600
            canvas_h = 500
        
        h, w = image.shape[:2]
        scale = min(canvas_w / w, canvas_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Изменяем размер изображения
        resized = cv2.resize(image, (new_w, new_h))
        
        # Конвертируем в PIL Image для tkinter
        pil_image = Image.fromarray(resized)
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # Очищаем canvas и отображаем новое изображение
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.photo, anchor=tk.CENTER)
        
        # Сохраняем параметры масштабирования
        self.scale_factor = scale
        self.canvas_offset_x = (canvas_w - new_w) // 2
        self.canvas_offset_y = (canvas_h - new_h) // 2
    def analyze_all_faces(self):
        if not self.face_bboxes or self.model is None:
            return

        self.update_status("Анализ всех лиц...")
        self.emotion_results = []

        # Очищаем поле результатов перед новым анализом
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)

        try:
            for i, (x, y, w, h) in enumerate(self.face_bboxes):
                face_roi = self.original_image[y:y+h, x:x+w]
                if face_roi.size == 0:
                    continue

                face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, (48, 48))
                face_input = face_resized.astype('float32') / 255.0
                face_input = np.expand_dims(face_input, axis=(0, -1))

                pred = self.model.predict(face_input, verbose=0)[0]
                result = {
                    'face_index': i,
                    'coordinates': (x, y, w, h),
                    'emotions': {self.emotion_labels[j]: float(pred[j]) for j in range(len(self.emotion_labels))},
                    'dominant_emotion': self.emotion_labels[int(np.argmax(pred))],
                    'dominant_prob': float(np.max(pred)) * 100,
                    'analysis_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'image_path': self.image_path,
                    'id': str(uuid.uuid4())
                }
                self.emotion_results.append(result)
                self.display_results(result)  # Выводим результат

            if self.emotion_results:
                self.current_face_index = 0
                self.report_btn.config(state='normal')
            self.update_status(f"Анализ завершен: {len(self.emotion_results)} лиц")
            self.history.extend(self.emotion_results)
            with open('history.json', 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)

            # Автоматически создаём PDF-отчёт
            self.auto_generate_report()

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка анализа: {str(e)}")
            self.update_status("Ошибка анализа эмоций")

        finally:
            # Заблокировать поле после вывода
            self.results_text.config(state='disabled')
    def auto_generate_report(self):
        """Автоматически создаёт PDF-отчёт без запроса пути"""
        if not self.emotion_results:
            return

        # Генерируем имя файла на основе даты и имени изображения
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{os.path.splitext(os.path.basename(self.image_path))[0]}_{timestamp}.pdf"
        file_path = os.path.join('reports', filename)

        os.makedirs('reports', exist_ok=True)

        try:
            self.update_status("Создание PDF отчета...")

            # Единый размер для всех страниц: горизонтальный А4
            FIG_WIDTH = 11.69   # дюймы
            FIG_HEIGHT = 8.27   # дюймы
            DPI = 150           # достаточно для качественной печати и отображения

            emotions_list = ['Удивление', 'Страх', 'Отвращение', 'Радость', 'Грусть', 'Злость', 'Безразличие']
            colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#95a5a6', '#3498db', '#9b59b6']

            with PdfPages(file_path) as pdf:
                def save_figure_horizontal(fig):
                    pdf.savefig(fig, bbox_inches=None, pad_inches=0, dpi=300)
                    plt.close(fig)

                # === 1. Титульная страница ===
                fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
                plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15)
                ax.axis('off')
                ax.text(0.5, 0.6, 'MoodLens - Отчет анализа эмоций', fontsize=20, fontweight='bold', ha='center')
                ax.text(0.5, 0.4, f'Дата: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', fontsize=14, ha='center')
                ax.text(0.5, 0.3, f'Изображение: {os.path.basename(self.image_path)}', fontsize=12, ha='center')
                save_figure_horizontal(fig)

                fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
                plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15)
                ax.axis('off')

                img_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                if img_rgb.dtype != np.uint8:
                    img_rgb = img_rgb.astype(np.uint8)

                img_h, img_w = img_rgb.shape[:2]

                max_img_width_inch = 0.6 * FIG_WIDTH
                max_img_height_inch = 0.5 * FIG_HEIGHT

                max_img_width_px = int(max_img_width_inch * 300)
                max_img_height_px = int(max_img_height_inch * 300)

                scale = min(1.0, max_img_width_px / img_w, max_img_height_px / img_h)
                new_w = int(img_w * scale)
                new_h = int(img_h * scale)

                from PIL import Image as PILImage
                pil_img = PILImage.fromarray(img_rgb)
                pil_img = pil_img.resize((new_w, new_h), PILImage.LANCZOS)
                img_display = np.array(pil_img)

                ax.imshow(img_display, interpolation='none', cmap='gray')
                ax.set_title(f'Исходное изображение\n({len(self.face_bboxes)} лиц обнаружено)', 
                            fontsize=14, pad=20, ha='center')

                for i, (x, y, w, h) in enumerate(self.face_bboxes):
                    x_scaled = x * scale
                    y_scaled = y * scale
                    w_scaled = w * scale
                    h_scaled = h * scale
                    
                    rect = plt.Rectangle((x_scaled, y_scaled), w_scaled, h_scaled, 
                                        fill=False, color='red', linewidth=2)
                    ax.add_patch(rect)
                    ax.text(x_scaled, y_scaled - 8, f'Face {i+1}', color='red', fontsize=10,
                            bbox=dict(facecolor='white', alpha=0.8))

                ax.set_xticks([])
                ax.set_yticks([])

                save_figure_horizontal(fig)

                # Графики по каждому лицу ===
                for result in self.emotion_results:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))
                    plt.subplots_adjust(left=0.08, right=0.95, top=0.85, bottom=0.15, wspace=0.3)

                    sorted_emotions = sorted(result['emotions'].items(), key=lambda x: x[1], reverse=True)
                    sorted_labels = [item[0] for item in sorted_emotions]
                    sorted_probs = [item[1] * 100 for item in sorted_emotions]
                    sorted_colors = [colors[emotions_list.index(label)] for label in sorted_labels]

                    bars = ax1.bar(sorted_labels, sorted_probs, color=sorted_colors, alpha=0.7, width=0.6)
                    ax1.set_title(f'Гистограмма для лица {result["face_index"] + 1}', fontsize=16, pad=15)
                    ax1.set_ylabel('Вероятность (%)', fontsize=12)
                    ax1.set_ylim(0, 100)
                    ax1.tick_params(axis='x', labelsize=10, rotation=45)
                    ax1.tick_params(axis='y', labelsize=10)
                    ax1.set_xticklabels(sorted_labels, rotation=45, ha='right')
                    for bar, prob in zip(bars, sorted_probs):
                        ax1.text(bar.get_x() + bar.get_width()/2., min(prob + 2, 98),
                                f'{prob:.1f}%', ha='center', va='bottom', fontsize=9)

                    ax2.axis('off')
                    table_data = [[emotion, f"{prob*100:.1f}%"] for emotion, prob in sorted_emotions]
                    table = ax2.table(cellText=table_data,
                                    colLabels=['Эмоция', 'Вероятность'],
                                    cellLoc='center',
                                    loc='center',
                                    bbox=[0.1, 0.1, 0.8, 0.8])
                    table.auto_set_font_size(False)
                    table.set_fontsize(10)
                    table.scale(1, 1.5)
                    ax2.set_title('Детальная статистика', fontsize=16, pad=15)

                    save_figure_horizontal(fig)

            self.update_status(f"Отчет сохранен: {file_path}")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка создания отчета: {str(e)}")
            self.update_status("Ошибка создания отчета")
    def display_results(self, result):
        # Включаем на запись
        self.results_text.config(state='normal')

        self.results_text.insert(tk.END, f"Анализ лица {result['face_index'] + 1}\n")
        self.results_text.insert(tk.END, "\n")
        self.results_text.insert(tk.END, f"Доминирующая эмоция: {result['dominant_emotion']}\n")
        self.results_text.insert(tk.END, f"Уверенность: {result['dominant_prob']:.1f}%\n\n")
        self.results_text.insert(tk.END, "Распределение эмоций:\n")

        # Находим максимальную длину метки (до двоеточия)
        max_label_len = max(len(emotion) for emotion in result['emotions'].keys())
        
        # Сортируем эмоции по убыванию вероятности
        sorted_emotions = sorted(result['emotions'].items(), key=lambda x: x[1], reverse=True)

        for emotion, prob in sorted_emotions:
            prob_str = f"{prob*100:5.1f}"  # 5.1f → " 83.8" (5 символов)
            bar = "█" * int(prob * 100 / 5)

            # Формируем строку с выравниванием
            label_part = f"{emotion}:"
            padding = " " * (max_label_len - len(emotion) + 1)  # +1 для пробела после двоеточия

            self.results_text.insert(tk.END, f"{label_part}{padding}{prob_str}% {bar}\n")

        self.results_text.insert(tk.END, "\n")

        # Опять блокируем
        self.results_text.config(state='disabled')
    def generate_report(self):
        if not self.emotion_results:
            messagebox.showwarning("Предупреждение", "Нет данных для отчета")
            return

        file_path = filedialog.asksaveasfilename(
            title="Сохранить отчет PDF",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")]
        )
        if not file_path:
            return

        try:
            self.update_status("Создание PDF отчета...")

            # Единый размер для всех страниц: горизонтальный А4
            FIG_WIDTH = 11.69   # дюймы
            FIG_HEIGHT = 8.27   # дюймы
            DPI = 150           # достаточно для качественной печати и отображения

            emotions_list = ['Удивление', 'Страх', 'Отвращение', 'Радость', 'Грусть', 'Злость', 'Безразличие']
            colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#95a5a6', '#3498db', '#9b59b6']

            with PdfPages(file_path) as pdf:
                def save_figure_horizontal(fig):
                    # ВАЖНО: bbox_inches=None и pad_inches=0 — чтобы сохранить строго заданный размер
                    pdf.savefig(fig, bbox_inches=None, pad_inches=0, dpi=300)  # или 200, 150 — по желанию                    plt.close(fig)

                # === 1. Титульная страница ===
                fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
                plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15)
                ax.axis('off')
                ax.text(0.5, 0.6, 'MoodLens - Отчет анализа эмоций', fontsize=20, fontweight='bold', ha='center')
                ax.text(0.5, 0.4, f'Дата: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', fontsize=14, ha='center')
                ax.text(0.5, 0.3, f'Изображение: {os.path.basename(self.image_path)}', fontsize=12, ha='center')
                save_figure_horizontal(fig)

                            # Страница с изображением 
                fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
                plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15)
                ax.axis('off')

                # Подготовка изображения
                img_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                if img_rgb.dtype != np.uint8:
                    img_rgb = img_rgb.astype(np.uint8)

                img_h, img_w = img_rgb.shape[:2]

                # Определяем максимальные размеры для отображения (в дюймах)
                max_img_width_inch = 0.6 * FIG_WIDTH   # 60% ширины страницы
                max_img_height_inch = 0.5 * FIG_HEIGHT # 50% высоты страницы

                # Переводим в пиксели при DPI=300 (для сохранения качества)
                max_img_width_px = int(max_img_width_inch * 300)
                max_img_height_px = int(max_img_height_inch * 300)

                # Масштабируем только если нужно УМЕНЬШИТЬ (не увеличиваем!)
                scale = min(1.0, max_img_width_px / img_w, max_img_height_px / img_h)
                new_w = int(img_w * scale)
                new_h = int(img_h * scale)

                # Качественное масштабирование через PIL
                from PIL import Image as PILImage
                pil_img = PILImage.fromarray(img_rgb)
                pil_img = pil_img.resize((new_w, new_h), PILImage.LANCZOS)
                img_display = np.array(pil_img)

                ax.imshow(img_display, interpolation='none', cmap='gray')

                ax.set_title(f'Исходное изображение\n', 
                            fontsize=14, pad=20, ha='center')

                # Рисуем bounding boxes
                for i, (x, y, w, h) in enumerate(self.face_bboxes):
                    x_scaled = x * scale
                    y_scaled = y * scale
                    w_scaled = w * scale
                    h_scaled = h * scale
                    
                    rect = plt.Rectangle((x_scaled, y_scaled), w_scaled, h_scaled, 
                                        fill=False, color='red', linewidth=2)
                    ax.add_patch(rect)
                    ax.text(x_scaled, y_scaled - 8, f'Face {i+1}', color='red', fontsize=10,
                            bbox=dict(facecolor='white', alpha=0.8))

                # Убираем оси
                ax.set_xticks([])
                ax.set_yticks([])

                save_figure_horizontal(fig)
                # Графики по каждому лицу ===
                for result in self.emotion_results:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))
                    plt.subplots_adjust(left=0.08, right=0.95, top=0.85, bottom=0.15, wspace=0.3)

                    sorted_emotions = sorted(result['emotions'].items(), key=lambda x: x[1], reverse=True)
                    sorted_labels = [item[0] for item in sorted_emotions]
                    sorted_probs = [item[1] * 100 for item in sorted_emotions]
                    sorted_colors = [colors[emotions_list.index(label)] for label in sorted_labels]

                    # Столбчатая диаграмма
                    bars = ax1.bar(sorted_labels, sorted_probs, color=sorted_colors, alpha=0.7, width=0.6)
                    ax1.set_title(f'Гистограмма для лица {result["face_index"] + 1}', fontsize=16, pad=15)
                    ax1.set_ylabel('Вероятность (%)', fontsize=12)
                    ax1.set_ylim(0, 100)
                    ax1.tick_params(axis='x', labelsize=10, rotation=45)
                    ax1.tick_params(axis='y', labelsize=10)
                    ax1.set_xticklabels(sorted_labels, rotation=45, ha='right')
                    for bar, prob in zip(bars, sorted_probs):
                        ax1.text(bar.get_x() + bar.get_width()/2., min(prob + 2, 98),
                                f'{prob:.1f}%', ha='center', va='bottom', fontsize=9)

                    # Таблица
                    ax2.axis('off')
                    table_data = [[emotion, f"{prob*100:.1f}%"] for emotion, prob in sorted_emotions]
                    table = ax2.table(cellText=table_data,
                                    colLabels=['Эмоция', 'Вероятность'],
                                    cellLoc='center',
                                    loc='center',
                                    bbox=[0.1, 0.1, 0.8, 0.8])
                    table.auto_set_font_size(False)
                    table.set_fontsize(10)
                    table.scale(1, 1.5)
                    ax2.set_title('Детальная статистика', fontsize=16, pad=15)

                    save_figure_horizontal(fig)

            self.update_status(f"Отчет сохранен: {file_path}")
            messagebox.showinfo("Успех", f"PDF отчет сохранен!\n{file_path}")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка создания отчета: {str(e)}")
            self.update_status("Ошибка создания отчета")
def main():
    os.makedirs('reports', exist_ok=True)
    root = tk.Tk()
    app = EmotionRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()