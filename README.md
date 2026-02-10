## ♻️ Garbage Classification System
### Transfer Learning with MobileNetV2 + Tkinter Desktop App

This repository contains a complete **end‑to‑end garbage classification system**:

- **Model training notebook** (`garbadgeclassification12c.ipynb`) that fine‑tunes **MobileNetV2** on a multi‑class garbage / recycling dataset.  
- **Exported Keras model** (`garbage.keras`) ready for inference.  
- **Desktop application** (`interface.py`) built with **Tkinter** to classify images of waste in a clean fullscreen UI.

The goal is to make **waste sorting easier** and support **recycling / smart‑bin applications** with a simple, local, no‑cloud dependency workflow.

---

## 🧱 Repository Structure

- **`garbadgeclassification12c.ipynb`**  
  Jupyter notebook used to **train and evaluate** the MobileNetV2 model:
  - Loads and preprocesses the garbage dataset  
  - Applies data augmentation  
  - Fine‑tunes MobileNetV2 on 12 classes  
  - Tracks metrics and saves the **best model** to `garbage.keras`

- **`garbage.keras`**  
  Serialized Keras model (TensorFlow 2) that is **loaded by the desktop app for inference**.  
  It must be kept **next to** `interface.py` so it can be found by `tf.keras.models.load_model("garbage.keras")`.

- **`interface.py`**  
  Tkinter‑based **GUI application**:
  - Loads `garbage.keras` at startup  
  - Lets the user pick a local image  
  - Preprocesses the image exactly like during training (MobileNetV2 `preprocess_input`, resize to \(224 \times 224\))  
  - Runs the model and displays **predicted class + confidence**  
  - Provides **Reset** and **Exit** controls  
  - Runs in **fullscreen** with keyboard shortcuts

- **`README.md`**  
  Documentation for the project (you are reading it).

---

## 🔍 Model & Classes

The model is a **MobileNetV2** network fine‑tuned on **12 garbage categories**.  
The exact class order is stored in `interface.py` and is used both during training and inference:

1. battery  
2. biological  
3. brown-glass  
4. cardboard  
5. clothes  
6. green-glass  
7. metal  
8. paper  
9. plastic  
10. shoes  
11. trash  
12. white-glass  

The interface uses this list to convert the model’s raw logits / probabilities into a **human‑readable label**.

---

## 🖥 Desktop Interface Overview (`interface.py`)

**Tech stack**
- **Python**, **Tkinter** for the GUI  
- **TensorFlow / Keras** for loading and running the model  
- **Pillow (PIL)** for image loading & display  
- **NumPy** for array handling

**Main features**
- **Fullscreen UI** with dark theme  
- **Load Image** button (JPEG/PNG)  
- **Predict** button that:
  - Preprocesses the image to \(224 \times 224\) with `preprocess_input`
  - Feeds it into `garbage.keras`
  - Finds the class with `np.argmax` and its confidence `np.max`
  - Displays: `Prediction: <class_name> (<confidence>)`
- **Reset** button to clear image and text  
- **Exit** button to close the app  
- Keyboard shortcuts:
  - `Esc` → exit fullscreen  
  - `F11` → toggle fullscreen

---

## 📦 Requirements

You need a working Python 3 environment and the following Python packages:

- `tensorflow` (2.x, with Keras included)  
- `numpy`  
- `Pillow`  
- `tkinter` (usually comes with standard Python on Windows; on some systems it may need to be installed separately)

You can install the main dependencies with:

```bash
pip install tensorflow pillow numpy
```

> **Note**: TensorFlow has platform‑specific builds; if you encounter issues, follow the official TensorFlow installation guide for your OS and Python version.

---

## ⚙️ How to Run the App

1. **Clone or download** this repository.
2. Ensure that **`garbage.keras`** and **`interface.py`** are in the **same folder**.
3. Install the dependencies (see **Requirements** section).
4. From the project directory, run:

```bash
python interface.py
```

5. The window will open in **fullscreen**:
   - Click **“Load Image”** and choose a local image of waste.  
   - Click **“Predict”** to see the predicted class and confidence.  
   - Use **“Reset”** to clear the UI or **“Exit”** to close the application.  
   - Press **Esc** to quickly exit fullscreen, **F11** to toggle it.

---

## 📚 Training the Model (Notebook)

The notebook `garbadgeclassification12c.ipynb` is designed for:

- Loading your garbage classification dataset
- Splitting into train / validation (and possibly test) sets
- Building a **MobileNetV2** transfer learning pipeline
- Applying augmentations and regularization
- Training and monitoring accuracy / loss
- Saving the **best performing weights** to `garbage.keras`

If you change:
- The dataset  
- The number of classes  
- Image size or preprocessing  

…make sure to:
1. Retrain via the notebook.  
2. Export a new `garbage.keras`.  
3. Update the `classes` list and any preprocessing constants in `interface.py` to match the new model.

---

## 🧪 Tips & Best Practices

- Always ensure that your inference preprocessing (resize, color mode, `preprocess_input`) is **identical** to training.
- Keep `garbage.keras` in sync with the version of TensorFlow you use to train it.
- For best results, feed images that are **well‑lit** and show the object clearly.

---

## ✅ Summary

This repository gives you:

- A **training notebook** to build a MobileNetV2‑based garbage classifier.  
- A **saved Keras model** (`garbage.keras`) ready for deployment.  
- A **Tkinter desktop interface** that makes garbage classification accessible to non‑technical users in a single click.

You can extend it with new classes, improve the dataset, or adapt the interface to integrate with larger recycling or smart‑city systems.
