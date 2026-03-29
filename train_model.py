"""
train_model.py
--------------
Downloads the Face Mask Dataset and trains a MobileNetV2-based
binary classifier (with_mask vs without_mask).

Run once before starting the app:
    python train_model.py
"""

import os
import urllib.request
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# ── Config ─────────────────────────────────────────────────────────────────────
INIT_LR     = 1e-4
EPOCHS      = 20
BATCH_SIZE  = 32
IMG_SIZE    = 224
MODEL_DIR   = "model"
DATA_DIR    = "dataset"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR,  exist_ok=True)

# ── 1. Download Dataset ────────────────────────────────────────────────────────
DATASET_URL = "https://github.com/chandrikadeb7/Face-Mask-Detection/archive/refs/heads/master.zip"
ZIP_PATH    = os.path.join(DATA_DIR, "face-mask.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "face-mask-extracted")
IMAGES_DIR  = os.path.join(EXTRACT_DIR, "Face-Mask-Detection-master", "dataset")

print("\n╔══════════════════════════════════════════╗")
print("║    Face Mask Detection — Model Trainer  ║")
print("╚══════════════════════════════════════════╝\n")

if not os.path.exists(IMAGES_DIR):
    print("[1/6] Downloading Face Mask Dataset (~45MB)...")
    urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
    print("      Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)
    print("      Done!")
else:
    print("[1/6] Dataset already present, skipping download.")

# ── 2. Load Images ─────────────────────────────────────────────────────────────
print("[2/6] Loading and preprocessing images...")

data   = []
labels = []

CATEGORIES = ["with_mask", "without_mask"]

for category in CATEGORIES:
    path = os.path.join(IMAGES_DIR, category)
    if not os.path.exists(path):
        print(f"      WARNING: {path} not found, skipping.")
        continue
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        try:
            image = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
            image = img_to_array(image)
            image = preprocess_input(image)
            data.append(image)
            labels.append(category)
        except Exception:
            pass

print(f"      Total images loaded : {len(data)}")

# ── 3. Encode Labels ───────────────────────────────────────────────────────────
print("[3/6] Encoding labels...")

lb     = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data   = np.array(data, dtype="float32")
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.20, stratify=labels, random_state=42
)

print(f"      Train: {len(X_train)} | Test: {len(X_test)}")
print(f"      Classes: {lb.classes_}")

# ── 4. Build Model ─────────────────────────────────────────────────────────────
print("[4/6] Building MobileNetV2 transfer learning model...")

baseModel = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3))
)

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

from tensorflow.keras.optimizers.schedules import ExponentialDecay
lr_schedule = ExponentialDecay(
    initial_learning_rate=INIT_LR,
    decay_steps=len(X_train) // BATCH_SIZE,
    decay_rate=0.96
)
opt = Adam(learning_rate=lr_schedule)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print(f"      Model built! Parameters: {model.count_params():,}")

# ── 5. Train ───────────────────────────────────────────────────────────────────
print(f"[5/6] Training for {EPOCHS} epochs (this takes ~5-10 mins)...")

aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

history = model.fit(
    aug.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    validation_data=(X_test, y_test),
    validation_steps=len(X_test) // BATCH_SIZE,
    epochs=EPOCHS
)

# ── 6. Evaluate & Save ─────────────────────────────────────────────────────────
print("[6/6] Evaluating and saving model...")

predIdxs = model.predict(X_test, batch_size=BATCH_SIZE)
predIdxs = np.argmax(predIdxs, axis=1)

print("\n── Classification Report ───────────────────")
print(classification_report(
    np.argmax(y_test, axis=1),
    predIdxs,
    target_names=lb.classes_
))

final_acc = history.history["val_accuracy"][-1]
final_loss = history.history["val_loss"][-1]

metrics = {
    "accuracy":   round(float(final_acc) * 100, 2),
    "loss":       round(float(final_loss), 4),
    "epochs":     EPOCHS,
    "train_size": len(X_train),
    "test_size":  len(X_test),
    "classes":    list(lb.classes_),
    "train_acc_history":  [round(float(a)*100,2) for a in history.history["accuracy"]],
    "val_acc_history":    [round(float(a)*100,2) for a in history.history["val_accuracy"]],
}

model.save(os.path.join(MODEL_DIR, "mask_detector.keras"))

with open(os.path.join(MODEL_DIR, "label_binarizer.pkl"), "wb") as f:
    pickle.dump(lb, f)

with open(os.path.join(MODEL_DIR, "metrics.pkl"), "wb") as f:
    pickle.dump(metrics, f)

print(f"\n✅ Model saved to model/mask_detector.keras")
print(f"   Validation Accuracy : {metrics['accuracy']}%")
print(f"\nNow run:  python app.py")
