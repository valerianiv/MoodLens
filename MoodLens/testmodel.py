import tkinter as tk
from tkinter import filedialog, messagebox

root = tk.Tk()
root.title("Тест GUI")
root.geometry("300x200")

def test_file_dialog():
    file_path = filedialog.askopenfilename(
        title="Выберите изображение",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if file_path:
        messagebox.showinfo("Успех", f"Выбран файл: {file_path}")
    else:
        messagebox.showinfo("Инфо", "Файл не выбран")

btn = tk.Button(root, text="Тест проводника", command=test_file_dialog)
btn.pack(pady=20)

root.mainloop()