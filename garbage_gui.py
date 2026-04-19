import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# The 12 classification classes derived from the training set
CLASS_NAMES = [
    'battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
    'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass'
]

class GarbageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Garbage Classification AI")
        self.root.geometry("600x700")
        self.root.configure(bg="#2c3e50")
        self.root.resizable(False, False)
        
        # Load the pre-trained Keras model
        try:
            self.model = tf.keras.models.load_model("garbage.keras")
        except Exception as e:
            messagebox.showerror("Model Error", f"Could not load 'garbage.keras'.\nError: {e}")
            self.model = None

        self.image_path = None
        self.img_label = None

        self.setup_ui()

    def setup_ui(self):
        # ---------------- HEADER ---------------- #
        header_frame = tk.Frame(self.root, bg="#34495e", pady=20)
        header_frame.pack(fill=tk.X)
        
        title_label = tk.Label(
            header_frame, 
            text="Garbage Classification", 
            font=("Helvetica", 24, "bold"), 
            fg="#ecf0f1", 
            bg="#34495e"
        )
        title_label.pack()

        # ---------- IMAGE DISPLAY AREA ---------- #
        self.image_frame = tk.Frame(self.root, bg="#bdc3c7", width=350, height=350, bd=4, relief=tk.GROOVE)
        self.image_frame.pack(pady=30)
        self.image_frame.pack_propagate(False)  # Keep the frame fixed size
        
        self.placeholder_label = tk.Label(
            self.image_frame, 
            text="No Image Loaded", 
            font=("Helvetica", 14, "italic"), 
            fg="#7f8c8d", 
            bg="#bdc3c7"
        )
        self.placeholder_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # ------------ RESULT LABEL -------------- #
        self.result_label = tk.Label(
            self.root, 
            text="Awaiting Image...", 
            font=("Helvetica", 18, "bold"), 
            fg="#e74c3c", 
            bg="#2c3e50"
        )
        self.result_label.pack(pady=10)

        # -------------- BUTTONS ----------------- #
        btn_frame = tk.Frame(self.root, bg="#2c3e50")
        btn_frame.pack(pady=20)

        # Button styling dictionary
        button_style = {
            "font": ("Helvetica", 14, "bold"),
            "fg": "white",
            "width": 12,
            "bd": 0,
            "cursor": "hand2",
            "pady": 10
        }

        self.btn_load = tk.Button(
            btn_frame, text="Load Image", bg="#2980b9", activebackground="#3498db", 
            command=self.load_image, **button_style
        )
        self.btn_load.grid(row=0, column=0, padx=10)

        self.btn_predict = tk.Button(
            btn_frame, text="Predict", bg="#27ae60", activebackground="#2ecc71", 
            command=self.predict_image, **button_style
        )
        self.btn_predict.grid(row=0, column=1, padx=10)

        self.btn_reset = tk.Button(
            btn_frame, text="Reset", bg="#c0392b", activebackground="#e74c3c", 
            command=self.reset_image, **button_style
        )
        self.btn_reset.grid(row=0, column=2, padx=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)
            self.result_label.config(text="Ready to Predict", fg="#f1c40f")

    def display_image(self, path):
        img = Image.open(path)
        # Resize for display purposes on the UI
        img = img.resize((350, 350), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(img)
        
        if self.img_label is None:
            self.img_label = tk.Label(self.image_frame, image=self.tk_img, bg="#bdc3c7")
            self.img_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        else:
            self.img_label.config(image=self.tk_img)
            
        self.placeholder_label.place_forget()

    def predict_image(self):
        if not self.image_path:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded!")
            return

        try:
            # 1. Load image and resize to MobileNetV2 input size (224x224)
            img = tf.keras.utils.load_img(self.image_path, target_size=(224, 224))
            # 2. Convert to Array
            img_array = tf.keras.utils.img_to_array(img)
            # 3. Apply MobileNetV2 specific preprocessing
            img_array = preprocess_input(img_array)
            # 4. Expand dimensions to shape (1, 224, 224, 3)
            img_array = np.expand_dims(img_array, axis=0)

            # 5. Make the prediction
            predictions = self.model.predict(img_array)
            predicted_index = np.argmax(predictions[0])
            confidence = predictions[0][predicted_index] * 100
            predicted_class = CLASS_NAMES[predicted_index].upper()

            # Update the Result label
            self.result_label.config(text=f"Prediction: {predicted_class} ({confidence:.2f}%)", fg="#2ecc71")
            
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred during prediction:\n{e}")

    def reset_image(self):
        self.image_path = None
        if self.img_label:
            self.img_label.destroy()
            self.img_label = None
        self.placeholder_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.result_label.config(text="Awaiting Image...", fg="#e74c3c")

if __name__ == "__main__":
    root = tk.Tk()
    app = GarbageClassifierApp(root)
    root.mainloop()