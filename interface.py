import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ======================================
# LOAD MODEL
# ======================================
print("Loading model...")
model = tf.keras.models.load_model(
    "garbage.keras"
)
print("Model loaded.")

IMG_SIZE = 224

# ======================================
# EXACT TRAINING CLASS ORDER
# ======================================
classes = [
    "battery",
    "biological",
    "brown-glass",
    "cardboard",
    "clothes",
    "green-glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
    "white-glass"
]

print("Classes:", classes)

# ======================================
# IMAGE PREPROCESS
# ======================================
def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    arr = np.array(img)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    return arr


# ======================================
# LOAD IMAGE
# ======================================
def load_image():
    global img_path, display_img

    path = filedialog.askopenfilename(
        filetypes=[("Images", "*.jpg *.png *.jpeg")]
    )

    if not path:
        return

    img_path = path

    img = Image.open(path)
    img.thumbnail((900, 650))

    display_img = ImageTk.PhotoImage(img)
    image_label.config(image=display_img)

    result_label.config(text="")


# ======================================
# PREDICT
# ======================================
def predict():
    if not img_path:
        messagebox.showinfo("Info", "Load image first.")
        return

    arr = preprocess_image(img_path)

    pred = model.predict(arr, verbose=0)
    idx = np.argmax(pred)
    conf = np.max(pred)

    result_label.config(
        text=f"Prediction: {classes[idx]}  ({conf:.2f})"
    )


# ======================================
# RESET
# ======================================
def reset():
    global img_path
    img_path = None
    image_label.config(image="")
    result_label.config(text="")


# ======================================
# FULLSCREEN CONTROL
# ======================================
def toggle_fullscreen(event=None):
    root.attributes("-fullscreen",
                    not root.attributes("-fullscreen"))


def exit_fullscreen(event=None):
    root.attributes("-fullscreen", False)


# ======================================
# UI WINDOW
# ======================================
root = tk.Tk()
root.title("Garbage Classification")
root.attributes("-fullscreen", True)

root.bind("<Escape>", exit_fullscreen)
root.bind("<F11>", toggle_fullscreen)

img_path = None
display_img = None

# Main layout
main = tk.Frame(root, bg="#1e1e1e")
main.pack(fill="both", expand=True)

# Title
tk.Label(
    main,
    text="Garbage Classification System",
    font=("Arial", 26, "bold"),
    fg="white",
    bg="#1e1e1e"
).pack(pady=10)

# Content area
content = tk.Frame(main, bg="#1e1e1e")
content.pack(fill="both", expand=True)

# Image display area
image_frame = tk.Frame(content, bg="#1e1e1e")
image_frame.pack(side="left", fill="both", expand=True, padx=20)

image_label = tk.Label(image_frame, bg="#1e1e1e")
image_label.pack(expand=True)

# Control panel
control_frame = tk.Frame(content, bg="#2c3e50", width=260)
control_frame.pack(side="right", fill="y")

btn_style = {
    "font": ("Arial", 16),
    "width": 16,
    "fg": "white",
    "pady": 10
}

tk.Button(control_frame, text="Load Image",
          command=load_image,
          bg="#3498db", **btn_style).pack(pady=15)

tk.Button(control_frame, text="Predict",
          command=predict,
          bg="#2ecc71", **btn_style).pack(pady=15)

tk.Button(control_frame, text="Reset",
          command=reset,
          bg="#f39c12", **btn_style).pack(pady=15)

tk.Button(control_frame, text="Exit",
          command=root.destroy,
          bg="#e74c3c", **btn_style).pack(pady=15)

# Result label
result_label = tk.Label(
    main,
    text="",
    font=("Arial", 20),
    fg="#00ffcc",
    bg="#1e1e1e"
)
result_label.pack(pady=10)

root.mainloop()
