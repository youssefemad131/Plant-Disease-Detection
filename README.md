# 🌿 Plant Disease Detection — CNN vs ANN Binary Classifier

A deep learning project that detects whether a plant leaf is **Healthy** or **Diseased** using two neural network architectures — a custom **CNN** and an **ANN** — trained on the **PlantVillage Dataset** and compared head-to-head.

> **Course:** Neural Networks
> **Team:** Norhan Medhat · Hemmat Hamdi · Noran Mohammed · Nada Mahmoud · Mina Saber · Yousef Emad
> **University:** Benha University — Computer Science, AI Track
> **Platform:** Google Colab (GPU: T4)

---

## 📌 Problem

Given a leaf image, classify it as:

| Label | Description |
|-------|-------------|
| 🌱 **Healthy** | No signs of disease |
| 😷 **Diseased** | Affected by one of 38 plant diseases |

---

## 📊 Dataset

**PlantVillage Dataset** — [Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

| Split | Images |
|-------|--------|
| Total Healthy | 15,083 |
| Total Diseased | 39,220 |
| After Balancing | 30,166 (15,083 each) |
| Train | ~68% |
| Validation | ~12% |
| Test | 20% |

> Dataset is **balanced** by undersampling the diseased class to match healthy count, preventing class bias.

**38 Plant Classes including:**
- Tomato (Late Blight, Early Blight, Bacterial Spot, Mosaic Virus...)
- Apple (Scab, Black Rot, Cedar Rust...)
- Grape, Potato, Corn, Pepper, Peach, Strawberry, and more

---

## 🏗️ Model Architectures

### CNN — Convolutional Neural Network (Primary Model)

Custom CNN built with `TensorFlow/Keras` for hierarchical spatial feature extraction:

```
Input: (128, 128, 3)
│
├── Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool → Dropout(0.25)
├── Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool → Dropout(0.25)
├── Conv2D(128) → BatchNorm → Conv2D(128) → MaxPool → Dropout(0.30)
│
├── GlobalAveragePooling2D
├── Dense(128, relu) → Dropout(0.5)
└── Dense(1, sigmoid)  ← Binary output
```

### ANN — Artificial Neural Network (Baseline Comparator)

Fully-connected network treating each pixel as an independent feature:

```
Input: (128, 128, 3) → Flatten → 49,152 features
│
├── Dense(512) → BatchNorm → Dropout(0.40)
├── Dense(256) → BatchNorm → Dropout(0.40)
├── Dense(128) → BatchNorm → Dropout(0.30)
├── Dense(64)  → Dropout(0.30)
└── Dense(1, sigmoid)
```

| Component | CNN | ANN |
|-----------|-----|-----|
| Optimizer | Adam | Adam (lr=0.001) |
| Loss | Binary Crossentropy | Binary Crossentropy |
| Input Size | 128 × 128 × 3 | 128 × 128 × 3 (flattened) |
| Output | Sigmoid | Sigmoid |
| Max Epochs | 30 | 30 |
| Batch Size | 32 | 32 |

> A **grid search** was run for the ANN over learning rates `[0.001, 0.0005, 0.0001]` and epoch counts `[5, 10]` to find the best starting configuration.

---

## ⚙️ Training Configuration

| Callback | Configuration |
|----------|--------------|
| `ModelCheckpoint` | Save best model by `val_accuracy` |
| `EarlyStopping` | Patience = 5, restore best weights |
| `ReduceLROnPlateau` | Monitor `val_loss`, factor=0.5, patience=3, min_lr=1e-6 |

---

## 📈 Results

| Metric | CNN | ANN |
|--------|-----|-----|
| Test Accuracy | **~98%+** | Lower (baseline) |
| Best Val Accuracy | **98.70%** | — |
| Training stopped at | Epoch 6 | Epoch 6 |

**CNN Training Progression:**

| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 1 | 90.31% | 86.41% |
| 2 | 95.75% | 95.52% |
| 4 | 97.61% | 97.54% |
| 6 | 98.13% | **98.70%** ✅ |

> The CNN significantly outperforms the ANN, demonstrating the value of convolutional feature extraction over raw pixel-wise classification.

---

## 💻 Code

### 📦 1. Imports

```python
import os
import glob as gb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import keras
import kagglehub
```

---

### 📥 2. Load Dataset

```python
path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")
color_path = os.path.join(path, "plantvillage dataset/color")

if not os.path.exists(color_path):
    raise FileNotFoundError(f"Color folder not found at {color_path}")
```

---

### 🗂️ 3. Build DataFrame & Balance Classes

```python
healthy_images, diseased_images = [], []
for folder in os.listdir(color_path):
    folder_path = os.path.join(color_path, folder)
    if os.path.isdir(folder_path):
        files = gb.glob(os.path.join(folder_path, '*.[jJ][pP][gG]'))
        if 'healthy' in folder.lower():
            healthy_images.extend(files)
        else:
            diseased_images.extend(files)

def get_label_from_path(img_path):
    folder_name = os.path.basename(os.path.dirname(img_path))
    return 'healthy' if 'healthy' in folder_name.lower() else 'diseased'

all_paths  = healthy_images + diseased_images
all_labels = [get_label_from_path(p) for p in all_paths]
df = pd.DataFrame({'image_path': all_paths, 'label': all_labels})

# Balance dataset (undersample diseased)
df_healthy  = df[df['label'] == 'healthy']
df_diseased = df[df['label'] == 'diseased']
min_size    = len(df_healthy)
df_diseased_balanced = df_diseased.sample(n=min_size, random_state=42)
df_balanced = pd.concat([df_healthy, df_diseased_balanced])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
```

---

### ✂️ 4. Train / Val / Test Split

```python
train_df, temp_df = train_test_split(df_balanced, test_size=0.2, random_state=42, stratify=df_balanced['label'])
val_df, test_df   = train_test_split(temp_df,     test_size=0.5, random_state=42, stratify=temp_df['label'])
```

---

### 🔄 5. Data Generators

```python
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen   = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(
    train_df, x_col="image_path", y_col="label",
    target_size=(128, 128), class_mode="binary", batch_size=32, shuffle=True
)
val_gen = val_datagen.flow_from_dataframe(
    val_df, x_col="image_path", y_col="label",
    target_size=(128, 128), class_mode="binary", batch_size=32, shuffle=False
)
test_gen = test_datagen.flow_from_dataframe(
    test_df, x_col="image_path", y_col="label",
    target_size=(128, 128), class_mode="binary", batch_size=32, shuffle=False
)
```

---

### 🏗️ 6. Build CNN Model

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dropout, Dense,
    GlobalAveragePooling2D, BatchNormalization
)

model = Sequential([
    # Block 1
    Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(128,128,3)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Block 2
    Conv2D(64, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Block 3
    Conv2D(128, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.30),

    # Classifier Head
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

---

### 🤖 7. Build ANN Model

```python
from tensorflow.keras.layers import Flatten

ann_model = Sequential([
    Flatten(input_shape=(128, 128, 3)),

    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.40),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.40),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.30),

    Dense(64, activation='relu'),
    Dropout(0.30),

    Dense(1, activation='sigmoid')
])

ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann_model.summary()
```

---

### 🏋️ 8. Train the Models

```python
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def get_callbacks(model_path):
    return [
        ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]

# Train CNN
history = model.fit(
    train_gen, validation_data=val_gen, epochs=30,
    callbacks=get_callbacks("/content/drive/MyDrive/best_cnn_model.h5")
)

# Train ANN
ann_history = ann_model.fit(
    train_gen, validation_data=val_gen, epochs=30,
    callbacks=get_callbacks("best_ann_model.h5")
)
```

---

### 📊 9. Plot Training Curves

```python
def plot_history(hist, title_prefix):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(hist.history['accuracy'],     label='Train')
    axes[0].plot(hist.history['val_accuracy'], label='Validation')
    axes[0].set_title(f'{title_prefix} — Accuracy')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(hist.history['loss'],     label='Train')
    axes[1].plot(hist.history['val_loss'], label='Validation')
    axes[1].set_title(f'{title_prefix} — Loss')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout(); plt.show()

plot_history(history,     "CNN")
plot_history(ann_history, "ANN")
```

---

### 🧪 10. Evaluate & Compare Models

```python
def full_evaluation(mdl, gen, label):
    loss, acc = mdl.evaluate(gen, verbose=0)
    print(f"\n=== {label} ===")
    print(f"Test Loss     : {loss:.4f}")
    print(f"Test Accuracy : {acc:.4f}")

    gen.reset()
    y_pred_prob = mdl.predict(gen).ravel()
    y_pred      = (y_pred_prob >= 0.5).astype(int)
    y_true      = gen.classes
    class_names = list(gen.class_indices.keys())

    print(classification_report(y_true, y_pred, target_names=class_names))

    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'{label} — Confusion Matrix (Test Set)')
    plt.tight_layout(); plt.show()

    return loss, acc

cnn_loss, cnn_acc = full_evaluation(model,     test_gen, "CNN")
ann_loss, ann_acc = full_evaluation(ann_model,  test_gen, "ANN")

print("\n=== Model Comparison on Test Set ===")
print(f"{'Model':<8} {'Loss':>8} {'Accuracy':>10}")
print("-" * 30)
print(f"{'CNN':<8} {cnn_loss:>8.4f} {cnn_acc:>10.4f}")
print(f"{'ANN':<8} {ann_loss:>8.4f} {ann_acc:>10.4f}")
```

---

### 🔍 11. Single Image Prediction

```python
def predict_and_show_image(image_path, mdl, model_name="Model"):
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized    = cv2.resize(img_rgb, (128, 128))
    img_normalized = img_resized / 255.0
    img_final      = np.expand_dims(img_normalized, axis=0)

    prediction = mdl.predict(img_final)[0][0]

    plt.imshow(img_rgb)
    if prediction > 0.5:
        plt.title(f"[{model_name}] 😷 Diseased  ({prediction:.2f})", color='red')
    else:
        plt.title(f"[{model_name}] 🌱 Healthy  ({1 - prediction:.2f})", color='green')
    plt.axis('off')
    plt.show()

predict_and_show_image("/content/your_leaf_image.jpg", model, "CNN")
```

---

### 🌐 12. Streamlit Web App

```python
# app.py
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model("best_cnn_model.h5")  # swap for best_ann_model.h5 to compare

st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image to check if it's healthy or diseased.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes     = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr        = cv2.imdecode(file_bytes, 1)
    img_rgb        = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

    img_resized    = cv2.resize(img_rgb, (128, 128))
    img_normalized = img_resized / 255.0
    img_final      = np.expand_dims(img_normalized, axis=0)

    prediction = model.predict(img_final)[0][0]

    if prediction > 0.5:
        st.error(f"😷 Prediction: Diseased  (confidence: {prediction:.2f})")
    else:
        st.success(f"🌱 Prediction: Healthy  (confidence: {1 - prediction:.2f})")
```

**Run locally:**
```bash
streamlit run app.py
```

---

## 📤 Sample Predictions

| Image | Prediction | Confidence |
|-------|------------|------------|
| Healthy Pepper Leaf | 🌱 Healthy | 0.97 |
| Bacterial Spot Pepper | 😷 Diseased | 0.94 |
| Tomato Leaf | 🌱 Healthy | 0.98 |

---

## 🛠️ Requirements

```bash
pip install tensorflow keras opencv-python pandas numpy matplotlib seaborn scikit-learn tqdm streamlit kagglehub
```

> **GPU recommended** — trained on Google Colab T4 GPU.

---

## 📁 Repository Structure

```
Plant-Disease-Detection/
│
├── PlantVillage_CNN.ipynb    ← Main notebook (EDA + CNN + ANN + Evaluation)
├── app.py                    ← Streamlit web app
├── best_cnn_model.h5         ← Saved CNN model
├── best_ann_model.h5         ← Saved ANN model
└── README.md
```

---

## 📚 Concepts Used

- `Convolutional Neural Networks (CNN)`
- `Artificial Neural Networks (ANN)`
- `Model Comparison & Benchmarking`
- `Batch Normalization & Dropout`
- `Data Balancing (Undersampling)`
- `ImageDataGenerators`
- `Binary Classification`
- `EarlyStopping & ReduceLROnPlateau`
- `Confusion Matrix & Classification Report`
- `Streamlit Deployment`

---

## 🔮 Future Work

- Extend to **multi-class** classification across all 38 disease categories
- Apply **data augmentation** (flips, rotations, zoom, colour jitter)
- Integrate **transfer learning** (EfficientNet, ResNet50, MobileNetV2)
- Deploy as a **mobile or web app** for on-field diagnosis
- Evaluate on **external field-captured images** to test real-world robustness
