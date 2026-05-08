# 🌿 Plant Disease Detection — CNN Binary Classifier

A deep learning project that detects whether a plant leaf is **Healthy** or **Diseased** using a custom **Convolutional Neural Network (CNN)** trained on the **PlantVillage Dataset**.

> **Course:** Neural Networks
> **Author:** Youssef Emad
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
| Train | 80% |
| Validation | 10% |
| Test | 10% |

> Dataset is **balanced** by undersampling the diseased class to match healthy count, preventing class bias.

**38 Plant Classes including:**
- Tomato (Late Blight, Early Blight, Bacterial Spot, Mosaic Virus...)
- Apple (Scab, Black Rot, Cedar Rust...)
- Grape, Potato, Corn, Pepper, Peach, Strawberry, and more

---

## 🏗️ Model Architecture

Custom CNN built with `TensorFlow/Keras`:

```
Input: (128, 128, 3)
│
├── Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool → Dropout(0.25)
├── Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool → Dropout(0.25)
├── Conv2D(128) → BatchNorm → Conv2D(128) → MaxPool → Dropout(0.25)
│
├── GlobalAveragePooling2D
├── Dense(256, relu) → Dropout(0.5)
└── Dense(1, sigmoid)  ← Binary output
```

| Component | Detail |
|-----------|--------|
| Optimizer | Adam |
| Loss | Binary Crossentropy |
| Input Size | 128 × 128 × 3 |
| Output | Sigmoid (0 = Healthy, 1 = Diseased) |
| Max Epochs | 30 |
| Batch Size | 32 |

---

## ⚙️ Training Configuration

| Callback | Configuration |
|----------|--------------|
| `ModelCheckpoint` | Save best model by `val_accuracy` |
| `EarlyStopping` | Patience = 5, restore best weights |
| `ReduceLROnPlateau` | Monitor `val_accuracy`, reduce on plateau |

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Best Val Accuracy | **98.70%** |
| Test Accuracy | **~98%+** |
| Training stopped at | Epoch 6 (best checkpoint) |

**Training progression:**

| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 1 | 90.31% | 86.41% |
| 2 | 95.75% | 95.52% |
| 4 | 97.61% | 97.54% |
| 6 | 98.13% | **98.70%** ✅ |

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
# Collect images by class
healthy_images, diseased_images = [], []
for folder in os.listdir(color_path):
    folder_path = os.path.join(color_path, folder)
    if os.path.isdir(folder_path):
        files = gb.glob(os.path.join(folder_path, '*.[jJ][pP][gG]'))
        if 'healthy' in folder.lower():
            healthy_images.extend(files)
        else:
            diseased_images.extend(files)

# Build DataFrame
def get_label_from_path(img_path):
    folder_name = os.path.basename(os.path.dirname(img_path))
    return 'healthy' if 'healthy' in folder_name.lower() else 'diseased'

all_paths  = healthy_images + diseased_images
all_labels = [get_label_from_path(p) for p in all_paths]
df = pd.DataFrame({'image_path': all_paths, 'label': all_labels})

# Balance dataset
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
train_df, temp_df   = train_test_split(df_balanced, test_size=0.2, random_state=42, stratify=df_balanced['label'])
val_df,   test_df   = train_test_split(temp_df,     test_size=0.5, random_state=42, stratify=temp_df['label'])
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
    Dropout(0.25),

    # Classifier Head
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

---

### 🏋️ 7. Train the Model

```python
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint(
    "/content/drive/MyDrive/best_model.h5",
    monitor='val_accuracy', save_best_only=True, verbose=1
)
early_stop = EarlyStopping(
    monitor='val_accuracy', patience=5, restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy', factor=0.5, patience=3, verbose=1
)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=[checkpoint, early_stop, reduce_lr]
)
```

---

### 📊 8. Plot Training Curves

```python
plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],     label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

---

### 🧪 9. Evaluate on Test Set

```python
loss, accuracy = model.evaluate(test_gen)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss:     {loss:.4f}")
```

---

### 🔍 10. Single Image Prediction

```python
def predict_and_show_image(image_path, model):
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized   = cv2.resize(img_rgb, (128, 128))
    img_normalized = img_resized / 255.0
    img_final     = np.expand_dims(img_normalized, axis=0)

    prediction = model.predict(img_final)[0][0]

    plt.imshow(img_rgb)
    if prediction > 0.5:
        plt.title(f"😷 Diseased  ({prediction:.2f})", color='red')
    else:
        plt.title(f"🌱 Healthy  ({1 - prediction:.2f})", color='green')
    plt.axis('off')
    plt.show()

# Example usage
predict_and_show_image("/content/your_leaf_image.jpg", model)
```

---

### 🌐 11. Streamlit Web App

```python
# app.py
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model("model.keras")

st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image to check if it's healthy or diseased.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes  = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr     = cv2.imdecode(file_bytes, 1)
    img_rgb     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
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
├── PlantVillage_CNN.ipynb    ← Main notebook (EDA + Training + Evaluation)
├── app.py                    ← Streamlit web app
├── model.keras               ← Saved trained model
└── README.md
```

---

## 📚 Concepts Used

- `Convolutional Neural Networks (CNN)`
- `Transfer Learning Ready Architecture`
- `Batch Normalization & Dropout`
- `Data Augmentation & Generators`
- `Binary Classification`
- `EarlyStopping & ReduceLROnPlateau`
- `Streamlit Deployment`
