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


class EmotionClassifier:
    def __init__(self):
        self.model_path = None
        self.input_size = (48, 48)
        self.emotion_categories = ['Удивление', 'Страх', 'Отвращение', 'Радость', 'Грусть', 'Злость', 'Безразличие']
        self.model = None
    
    def load_model(self, path: str):
        try:
            # Проверка максимальной длины пути
            if len(path) > 260:  # Максимальная длина пути в Windows
                raise ValueError("Слишком длинный путь к файлу модели")
                
            if os.path.exists(path):
                self.model = load_model(path)
                self.model_path = path
                return True
            else:
                print(f"Модель не найдена по пути: {path}")
                return False
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return False
    
    def preprocess_face(self, face_roi: np.ndarray) -> np.ndarray:
        if face_roi.size == 0:
            raise ValueError("Пустая область лица")
        
        # Проверка максимального размера входного изображения
        max_input_size = 10000  # 10000x10000 пикселей
        if face_roi.shape[0] > max_input_size or face_roi.shape[1] > max_input_size:
            raise ValueError(f"Размер изображения лица превышает максимально допустимый: {max_input_size}x{max_input_size}")
        
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, self.input_size)
        face_input = face_resized.astype('float32') / 255.0
        face_input = np.expand_dims(face_input, axis=(0, -1))
        
        return face_input
    
    def predict_emotion(self, face_data: np.ndarray) -> dict:
        if self.model is None:
            raise ValueError("Модель не загружена")
        
        # Проверка размера входных данных для модели
        expected_shape = (1, 48, 48, 1)
        if face_data.shape != expected_shape:
            raise ValueError(f"Неверный размер входных данных. Ожидается: {expected_shape}, получено: {face_data.shape}")
        
        pred = self.model.predict(face_data, verbose=0)[0]
        
        # Проверка корректности предсказаний
        if len(pred) != len(self.emotion_categories):
            raise ValueError("Количество предсказанных эмоций не соответствует ожидаемому")
        
        emotions = {self.emotion_categories[j]: float(pred[j]) for j in range(len(self.emotion_categories))}
        dominant_emotion, dominant_prob = self.get_dominant_emotion(pred)
        
        return {
            'emotions': emotions,
            'dominant_emotion': dominant_emotion,
            'dominant_prob': dominant_prob
        }
    
    def get_dominant_emotion(self, predictions: np.ndarray) -> tuple:
        # Проверка валидности индекса
        if len(predictions) == 0:
            raise ValueError("Пустой массив предсказаний")
            
        dominant_idx = int(np.argmax(predictions))
        
        # Проверка границ массива
        if dominant_idx < 0 or dominant_idx >= len(self.emotion_categories):
            raise ValueError(f"Некорректный индекс доминирующей эмоции: {dominant_idx}")
            
        dominant_emotion = self.emotion_categories[dominant_idx]
        dominant_prob = float(np.max(predictions)) * 100
        return dominant_emotion, dominant_prob


class FaceDetector:
    def __init__(self):
        self.face_detection_model = None
        self.detection_confidence = 0.3
        self.max_faces = 20  # Максимальное количество лиц для обработки
    
    def detect_faces(self, image: np.ndarray) -> list:
        # Проверка размера входного изображения
        if image.size == 0:
            raise ValueError("Пустое входное изображение")
            
        max_image_size = 10000  # 10000x10000 пикселей
        if image.shape[0] > max_image_size or image.shape[1] > max_image_size:
            raise ValueError(f"Размер изображения превышает максимально допустимый: {max_image_size}x{max_image_size}")
        
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        face_bboxes = []

        with face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=self.detection_confidence
        ) as fd:
            results = fd.process(rgb)

        if results.detections:
            # Ограничение максимального количества лиц
            faces_to_process = min(len(results.detections), self.max_faces)
            
            for i, detection in enumerate(results.detections):
                if i >= faces_to_process:
                    break  # Прекращаем обработку после достижения лимита
                    
                bbox = detection.location_data.relative_bounding_box
                x_min, y_min, width, height = self.calculate_face_coordinates(bbox, w, h)
                face_bboxes.append((x_min, y_min, width, height))

        return face_bboxes
    
    def draw_bounding_boxes(self, image: np.ndarray, boxes: list) -> np.ndarray:
        # Проверка количества bounding boxes
        if len(boxes) > self.max_faces:
            raise ValueError(f"Количество bounding boxes превышает максимально допустимое: {self.max_faces}")
            
        img_with_boxes = image.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                 (255, 0, 255), (0, 255, 255), (255, 165, 0)]

        for i, (x, y, w_box, h_box) in enumerate(boxes):
            # Проверка границ массива цветов
            color = colors[i % len(colors)]
            
            # Проверка координат bounding box
            if (x < 0 or y < 0 or x + w_box > image.shape[1] or y + h_box > image.shape[0]):
                continue  # Пропуск некорректных bounding boxes
                
            cv2.rectangle(img_with_boxes, (x, y), (x + w_box, y + h_box), color, 2)
            
            # Проверка длины текста
            face_text = f'Face {i+1}'
            if len(face_text) > 50:  # Ограничение длины текста
                face_text = face_text[:50]
                
            cv2.putText(img_with_boxes, face_text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return img_with_boxes
    
    def calculate_face_coordinates(self, bbox, image_width: int, image_height: int) -> tuple:
        # Проверка входных параметров
        if image_width <= 0 or image_height <= 0:
            raise ValueError("Некорректные размеры изображения")
            
        x_min = int(bbox.xmin * image_width)
        y_min = int(bbox.ymin * image_height)
        width = int(bbox.width * image_width)
        height = int(bbox.height * image_height)
        
        # Проверка корректности координат
        if x_min < 0 or y_min < 0 or width <= 0 or height <= 0:
            raise ValueError("Некорректные координаты лица")
            
        return x_min, y_min, width, height


class ImageProcessor:
    def __init__(self):
        self.scale_factor = 1.0
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0
        self.photo = None
    
    def display_image(self, image: np.ndarray, canvas: tk.Canvas):
        # Проверка размера изображения
        if image.size == 0:
            raise ValueError("Пустое изображение для отображения")
            
        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height()
        
        if canvas_w <= 0 or canvas_h <= 0:
            canvas_w, canvas_h = 600, 500
        
        h, w = image.shape[:2]
        
        # Ограничение максимального размера для отображения
        max_display_size = 4000  # 4000x4000 пикселей
        if w > max_display_size or h > max_display_size:
            scale_reduction = min(max_display_size / w, max_display_size / h)
            w = int(w * scale_reduction)
            h = int(h * scale_reduction)
            image = cv2.resize(image, (w, h))
        
        self.scale_factor = min(canvas_w / w, canvas_h / h)
        new_w, new_h = int(w * self.scale_factor), int(h * self.scale_factor)
        
        resized = self.resize_image(image, new_w, new_h)
        self.photo = self.convert_to_tkinter(resized)
        
        canvas.delete("all")
        canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.photo, anchor=tk.CENTER)
        
        self.canvas_offset_x = (canvas_w - new_w) // 2
        self.canvas_offset_y = (canvas_h - new_h) // 2
    
    def highlight_selected_face(self, image: np.ndarray, boxes: list, selected_index: int) -> np.ndarray:
        # Проверка индекса
        if selected_index < -1 or selected_index >= len(boxes):
            raise ValueError("Некорректный индекс выбранного лица")
            
        img_highlight = image.copy()
        for i, (x, y, w, h) in enumerate(boxes):
            color = (0, 255, 255) if i == selected_index else (255, 0, 0)
            thickness = 3 if i == selected_index else 2
            cv2.rectangle(img_highlight, (x, y), (x + w, y + h), color, thickness)
            
            # Проверка длины текста
            face_text = f'Face {i+1}'
            if len(face_text) > 50:
                face_text = face_text[:50]
                
            cv2.putText(img_highlight, face_text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5 if i != selected_index else 0.6, color, thickness)
        return img_highlight
    
    def on_canvas_click(self, event, boxes: list, scale_factor: float, offset_x: float, offset_y: float) -> int:
        x_click = (event.x - offset_x) / scale_factor
        y_click = (event.y - offset_y) / scale_factor

        for i, (x, y, w, h) in enumerate(boxes):
            if x <= x_click <= x + w and y <= y_click <= y + h:
                return i
        return -1
    
    def resize_image(self, image: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
        # Проверка размеров
        if new_w <= 0 or new_h <= 0:
            raise ValueError("Некорректные размеры для ресайза")
        return cv2.resize(image, (new_w, new_h))
    
    def convert_to_tkinter(self, image: np.ndarray) -> ImageTk.PhotoImage:
        pil_image = Image.fromarray(image)
        return ImageTk.PhotoImage(pil_image)


class ReportGenerator:
    def __init__(self):
        self.report_data = {}
        self.pdf_pages = None
        self.figure_size = (11.69, 8.27)
        self.colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#95a5a6', '#3498db', '#9b59b6']
        self.max_filename_length = 255  # Максимальная длина имени файла
    
    def create_title_page(self, image_path: str):
        # Проверка длины имени файла
        filename = os.path.basename(image_path)
        if len(filename) > self.max_filename_length:
            filename = filename[:self.max_filename_length] + "..."
            
        fig, ax = plt.subplots(figsize=self.figure_size)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15)
        ax.axis('off')
        ax.text(0.5, 0.6, 'MoodLens - Отчет анализа эмоций', fontsize=20, fontweight='bold', ha='center')
        ax.text(0.5, 0.4, f'Дата: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', fontsize=14, ha='center')
        ax.text(0.5, 0.3, f'Изображение: {filename}', fontsize=12, ha='center')
        return fig
    
    def add_image_page(self, image: np.ndarray, bboxes: list):
        fig, ax = plt.subplots(figsize=self.figure_size)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15)
        ax.axis('off')

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if img_rgb.dtype != np.uint8:
            img_rgb = img_rgb.astype(np.uint8)

        img_h, img_w = img_rgb.shape[:2]
        max_img_width_px = int(0.6 * self.figure_size[0] * 300)
        max_img_height_px = int(0.5 * self.figure_size[1] * 300)
        scale = min(1.0, max_img_width_px / img_w, max_img_height_px / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)

        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(img_rgb)
        pil_img = pil_img.resize((new_w, new_h), PILImage.LANCZOS)
        img_display = np.array(pil_img)

        ax.imshow(img_display, interpolation='none', cmap='gray')
        ax.set_title('Исходное изображение', fontsize=14, pad=20, ha='center')

        for i, (x, y, w, h) in enumerate(bboxes):
            x_scaled, y_scaled = x * scale, y * scale
            w_scaled, h_scaled = w * scale, h * scale
            
            # Проверка координат для отрисовки
            if (x_scaled >= 0 and y_scaled >= 0 and 
                x_scaled + w_scaled <= new_w and y_scaled + h_scaled <= new_h):
                
                rect = plt.Rectangle((x_scaled, y_scaled), w_scaled, h_scaled, 
                                   fill=False, color='red', linewidth=2)
                ax.add_patch(rect)
                
                # Проверка длины текста
                face_text = f'Face {i+1}'
                if len(face_text) > 20:
                    face_text = face_text[:20]
                    
                ax.text(x_scaled, y_scaled - 8, face_text, color='red', fontsize=10,
                       bbox=dict(facecolor='white', alpha=0.8))

        ax.set_xticks([])
        ax.set_yticks([])
        return fig
    
    def generate_table_data(self, emotion_data: dict) -> list:
        table_data = []
        for emotion, prob in emotion_data.items():
            # Проверка длины названия эмоции
            emotion_name = emotion
            if len(emotion_name) > 20:
                emotion_name = emotion_name[:20] + "..."
            table_data.append([emotion_name, f"{prob*100:.1f}%"])
        return table_data
    
    def add_emotion_charts(self, emotion_results: list):
        figures = []
        emotions_list = ['Удивление', 'Страх', 'Отвращение', 'Радость', 'Грусть', 'Злость', 'Безразличие']
        
        for result in emotion_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figure_size)
            plt.subplots_adjust(left=0.08, right=0.95, top=0.85, bottom=0.15, wspace=0.3)

            sorted_emotions = sorted(result['emotions'].items(), key=lambda x: x[1], reverse=True)
            sorted_labels = [item[0] for item in sorted_emotions]
            sorted_probs = [item[1] * 100 for item in sorted_emotions]
            sorted_colors = [self.colors[emotions_list.index(label)] for label in sorted_labels]

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
            table_data = self.generate_table_data(result['emotions'])
            table = ax2.table(cellText=table_data,
                            colLabels=['Эмоция', 'Вероятность'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0.1, 0.1, 0.8, 0.8])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            ax2.set_title('Детальная статистика', fontsize=16, pad=15)

            figures.append(fig)
        
        return figures
    
    def save_report(self, file_path: str, image_path: str, original_image: np.ndarray, 
                   face_bboxes: list, emotion_results: list):
        try:
            # Проверка длины пути для сохранения
            if len(file_path) > 260:  # Максимальная длина пути в Windows
                raise ValueError("Слишком длинный путь для сохранения отчета")
                
            with PdfPages(file_path) as pdf:
                def save_figure(fig):
                    pdf.savefig(fig, bbox_inches=None, pad_inches=0, dpi=300)
                    plt.close(fig)

                title_fig = self.create_title_page(image_path)
                save_figure(title_fig)

                image_fig = self.add_image_page(original_image, face_bboxes)
                save_figure(image_fig)

                emotion_figures = self.add_emotion_charts(emotion_results)
                for fig in emotion_figures:
                    save_figure(fig)

            return True
        except Exception as e:
            print(f"Ошибка сохранения отчета: {e}")
            return False


class EmotionRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MoodLens - Система распознавания эмоций")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2c3e50')

        self.emotion_classifier = EmotionClassifier()
        self.face_detector = FaceDetector()
        self.image_processor = ImageProcessor()
        self.report_generator = ReportGenerator()

        self.image_path = None
        self.original_image = None
        self.processed_image = None
        self.face_bboxes = []
        self.current_face_index = -1
        self.emotion_results = []

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
        model_path = os.path.join('models', 'simplecnn.h5')
        if not self.emotion_classifier.load_model(model_path):
            messagebox.showwarning("Предупреждение", "Модель классификации эмоций не найдена")

    def create_widgets(self):
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        title_frame = ttk.Frame(main_container)
        title_frame.pack(fill=tk.X, side=tk.TOP, pady=(0, 20))
        title_label = tk.Label(title_frame, text="MoodLens", 
                              font=("Consolas", 24, "bold"), 
                              fg="#3498db", bg='#2c3e50')
        title_label.pack(expand=True)

        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)
        content_frame.grid_rowconfigure(0, weight=0)
        content_frame.grid_rowconfigure(1, weight=1)
        content_frame.grid_columnconfigure(0, weight=1, uniform="equal")
        content_frame.grid_columnconfigure(1, weight=1, uniform="equal")

        upload_frame = ttk.LabelFrame(content_frame, text="Загрузка изображения", padding=15)
        upload_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        upload_buttons = ttk.Frame(upload_frame)
        upload_buttons.pack(fill=tk.BOTH, expand=True)
        upload_buttons.grid_columnconfigure(0, weight=1)
        upload_buttons.grid_rowconfigure(0, weight=1)
        self.upload_btn = ttk.Button(upload_buttons, text="Загрузить фото", command=self.load_image)
        self.upload_btn.grid(row=0, column=0, sticky="nsew", padx=5, pady=10)

        control_frame = ttk.LabelFrame(content_frame, text="Управление", padding=15)
        control_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=(0, 10))
        control_buttons = ttk.Frame(control_frame)
        control_buttons.pack(fill=tk.BOTH, expand=True)
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

        self.image_frame = ttk.LabelFrame(content_frame, text="Предпросмотр", padding=10)
        self.image_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10), pady=0)
        self.canvas = tk.Canvas(self.image_frame, bg='#34495e', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        results_frame = ttk.LabelFrame(content_frame, text="Результаты анализа", padding=10)
        results_frame.grid(row=1, column=1, sticky="nsew", padx=(10, 0), pady=0)
        self.results_text = tk.Text(results_frame, wrap=tk.WORD,
                                  bg='#ecf0f1', fg='#2c3e50', font=('Consolas', 10), state='disabled')
        scrollbar = ttk.Scrollbar(results_frame, command=self.results_text.yview)
        self.results_text.config(yscrollcommand=scrollbar.set)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.status_var = tk.StringVar(value="Готов к работе")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if not file_path:
            return

        # Проверка длины пути к файлу
        if len(file_path) > 260:
            messagebox.showerror("Ошибка", "Слишком длинный путь к файлу")
            return

        self.image_path = file_path
        self.face_bboxes = []
        self.emotion_results = []
        self.current_face_index = -1

        self.detect_btn.config(state='disabled')
        self.analyze_btn.config(state='disabled')
        self.report_btn.config(state='disabled')
        self.clear_results()

        try:
            from PIL import Image
            pil_img = Image.open(file_path).convert('RGB')
            self.original_image = np.array(pil_img)[:, :, ::-1]

            if self.original_image is None or self.original_image.size == 0:
                raise ValueError("Изображение пустое или повреждено")

            # Проверка размера изображения
            h, w = self.original_image.shape[:2]
            max_image_size = 10000  # 10000x10000 пикселей
            if w > max_image_size or h > max_image_size:
                messagebox.showwarning("Предупреждение", 
                                     f"Изображение слишком большое. Будет уменьшено до {max_image_size}x{max_image_size}")
                scale = min(max_image_size / w, max_image_size / h)
                new_w, new_h = int(w * scale), int(h * scale)
                self.original_image = cv2.resize(self.original_image, (new_w, new_h))

            self.processed_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.image_processor.display_image(self.processed_image, self.canvas)

            self.update_status("Изображение загружено")
            self.detect_btn.config(state='normal')

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {str(e)}")
            self.update_status("Ошибка загрузки изображения")

    def detect_faces(self):
        if self.original_image is None:
            return

        self.update_status("Обнаружение лиц")
        try:
            self.face_bboxes = self.face_detector.detect_faces(self.original_image)

            if not self.face_bboxes:
                messagebox.showinfo("Информация", "Лица не обнаружены")
                self.update_status("Лица не обнаружены")
                return

            img_with_faces = self.face_detector.draw_bounding_boxes(self.processed_image, self.face_bboxes)
            self.image_processor.display_image(img_with_faces, self.canvas)

            self.update_status(f"Обнаружено лиц: {len(self.face_bboxes)}")
            self.analyze_btn.config(state='normal')

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка обнаружения: {str(e)}")
            self.update_status("Ошибка обнаружения лиц")

    def analyze_all_faces(self):
        if not self.face_bboxes or self.emotion_classifier.model is None:
            return

        self.update_status("Анализ всех лиц")
        self.emotion_results = []
        self.clear_results()

        try:
            for i, (x, y, w, h) in enumerate(self.face_bboxes):
                # Проверка координат перед извлечением области лица
                if (x < 0 or y < 0 or x + w > self.original_image.shape[1] or 
                    y + h > self.original_image.shape[0]):
                    continue  # Пропуск некорректных bounding boxes
                    
                face_roi = self.original_image[y:y+h, x:x+w]
                if face_roi.size == 0:
                    continue

                face_input = self.emotion_classifier.preprocess_face(face_roi)
                emotion_prediction = self.emotion_classifier.predict_emotion(face_input)

                result = {
                    'face_index': i,
                    'coordinates': (x, y, w, h),
                    'emotions': emotion_prediction['emotions'],
                    'dominant_emotion': emotion_prediction['dominant_emotion'],
                    'dominant_prob': emotion_prediction['dominant_prob'],
                    'analysis_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'image_path': self.image_path,
                    'id': str(uuid.uuid4())
                }
                self.emotion_results.append(result)
                self.display_results(result)

            if self.emotion_results:
                self.current_face_index = 0
                self.report_btn.config(state='normal')

            self.update_status(f"Анализ завершен: {len(self.emotion_results)} лиц")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка анализа: {str(e)}")
            self.update_status("Ошибка анализа эмоций")

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

        # Проверка длины пути для сохранения
        if len(file_path) > 260:
            messagebox.showerror("Ошибка", "Слишком длинный путь для сохранения отчета")
            return

        self.update_status("Создание PDF отчета")

        if self.report_generator.save_report(file_path, self.image_path, 
                                           self.original_image, self.face_bboxes, 
                                           self.emotion_results):
            self.update_status(f"Отчет сохранен: {file_path}")
            messagebox.showinfo("Успех", f"PDF отчет сохранен: {file_path}")
        else:
            messagebox.showerror("Ошибка", "Не удалось создать отчет")
            self.update_status("Ошибка создания отчета")

    def on_canvas_click(self, event):
        if not self.face_bboxes:
            return

        selected_index = self.image_processor.on_canvas_click(
            event, self.face_bboxes, 
            self.image_processor.scale_factor,
            self.image_processor.canvas_offset_x,
            self.image_processor.canvas_offset_y
        )

        if selected_index != -1:
            self.current_face_index = selected_index
            img_highlight = self.image_processor.highlight_selected_face(
                self.processed_image, self.face_bboxes, self.current_face_index
            )
            self.image_processor.display_image(img_highlight, self.canvas)
            self.update_status(f"Выбрано лицо {self.current_face_index + 1}")

    def display_results(self, result):
        self.results_text.config(state='normal')
        
        self.results_text.insert(tk.END, f"Анализ лица {result['face_index'] + 1}\n\n")
        self.results_text.insert(tk.END, f"Доминирующая эмоция: {result['dominant_emotion']}\n")
        self.results_text.insert(tk.END, f"Уверенность: {result['dominant_prob']:.1f}%\n\n")
        self.results_text.insert(tk.END, "Распределение эмоций:\n")

        max_label_len = max(len(emotion) for emotion in result['emotions'].keys())
        sorted_emotions = sorted(result['emotions'].items(), key=lambda x: x[1], reverse=True)

        for emotion, prob in sorted_emotions:
            # Ограничение длины названия эмоции
            emotion_display = emotion
            if len(emotion_display) > 20:
                emotion_display = emotion_display[:20] + "..."
                
            prob_str = f"{prob*100:5.1f}"
            bar = "█" * int(prob * 100 / 5)
            label_part = f"{emotion_display}:"
            padding = " " * (max_label_len - len(emotion_display) + 1)
            self.results_text.insert(tk.END, f"{label_part}{padding}{prob_str}% {bar}\n")

        self.results_text.insert(tk.END, "\n")
        self.results_text.config(state='disabled')

    def clear_results(self):
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state='disabled')

    def update_status(self, message):
        # Ограничение длины статусного сообщения
        if len(message) > 100:
            message = message[:100] + "..."
        self.status_var.set(message)


def main():
    root = tk.Tk()
    app = EmotionRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()