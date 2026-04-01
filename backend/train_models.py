"""
Fashion Paglu — Model Training
2 Deep Learning Models train karta hai:
1. Skin Tone Classifier (Neural Network)
2. Outfit Compatibility Scorer (Neural Network)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, json

os.makedirs("models",  exist_ok=True)
os.makedirs("reports", exist_ok=True)

print("="*55)
print("  Fashion Paglu — Deep Learning Model Training")
print("="*55)

# TensorFlow import
print("\n📦 Loading TensorFlow...")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.metrics          import classification_report, confusion_matrix
import seaborn as sns

print(f"   TensorFlow version: {tf.__version__}")
print(f"   GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# ══════════════════════════════════════════
# MODEL 1: SKIN TONE CLASSIFIER
# ══════════════════════════════════════════
print("\n" + "─"*55)
print("  MODEL 1: Skin Tone Classifier")
print("─"*55)

# Data load karo
skin_df = pd.read_csv("datasets/skin_tone_dataset.csv")
print(f"\n   📊 Dataset loaded: {len(skin_df)} samples")

# Features aur labels
X_skin = skin_df[[
    "r", "g", "b",
    "brightness", "r_g_diff",
    "r_b_diff",   "g_b_diff",
    "ita_angle"
]].values

y_skin = skin_df["skin_class"].values

# Train/Val/Test split
X_tr, X_te, y_tr, y_te = train_test_split(
    X_skin, y_skin, test_size=0.2, random_state=42, stratify=y_skin)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_tr, y_tr, test_size=0.15, random_state=42, stratify=y_tr)

print(f"   Train: {len(X_tr)} | Val: {len(X_val)} | Test: {len(X_te)}")

# Normalize karo
scaler_skin = StandardScaler()
X_tr  = scaler_skin.fit_transform(X_tr)
X_val = scaler_skin.transform(X_val)
X_te  = scaler_skin.transform(X_te)

# ── Model Architecture ──
skin_model = keras.Sequential([
    layers.Input(shape=(8,)),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu"),
    layers.Dense(6, activation="softmax")   # 6 skin tone classes
], name="SkinToneClassifier")

skin_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

skin_model.summary()

# Callbacks
callbacks_skin = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=8,
        restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=4, verbose=1),
    keras.callbacks.ModelCheckpoint(
        "models/skin_tone_model.h5",
        monitor="val_accuracy",
        save_best_only=True, verbose=0)
]

# Train!
print("\n   🚀 Training Skin Tone Model...")
history_skin = skin_model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=callbacks_skin,
    verbose=1
)

# Evaluate
test_loss, test_acc = skin_model.evaluate(X_te, y_te, verbose=0)
print(f"\n   ✅ Skin Tone Model:")
print(f"   Test Accuracy:  {test_acc*100:.2f}%")
print(f"   Test Loss:      {test_loss:.4f}")

# Classification report
y_pred_skin = np.argmax(skin_model.predict(X_te, verbose=0), axis=1)
class_names = ["Light", "Light-Med", "Medium",
               "Med-Dark", "Dark", "Deep"]
report_skin = classification_report(y_te, y_pred_skin,
                                     target_names=class_names)
print("\n   Classification Report:")
print(report_skin)

# Training plot save karo
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history_skin.history["accuracy"],     label="Train Acc")
axes[0].plot(history_skin.history["val_accuracy"], label="Val Acc")
axes[0].set_title("Skin Tone Model — Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(True)
axes[1].plot(history_skin.history["loss"],     label="Train Loss")
axes[1].plot(history_skin.history["val_loss"], label="Val Loss")
axes[1].set_title("Skin Tone Model — Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(True)
plt.tight_layout()
plt.savefig("reports/skin_tone_training.png", dpi=120)
plt.close()
print("   📊 Training graph: reports/skin_tone_training.png")

# Confusion matrix
cm = confusion_matrix(y_te, y_pred_skin)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_names, yticklabels=class_names,
            cmap="YlOrRd")
plt.title("Skin Tone — Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("reports/skin_tone_confusion_matrix.png", dpi=120)
plt.close()


# ══════════════════════════════════════════
# MODEL 2: OUTFIT COMPATIBILITY SCORER
# ══════════════════════════════════════════
print("\n" + "─"*55)
print("  MODEL 2: Outfit Compatibility Scorer")
print("─"*55)

outfit_df = pd.read_csv("datasets/outfit_compatibility_dataset.csv")
print(f"\n   📊 Dataset loaded: {len(outfit_df)} samples")

X_out = outfit_df[[
    "undertone_enc", "body_type_enc", "occasion_enc",
    "n_items", "has_outerwear", "has_accessory",
    "color_contrast", "pattern_mix"
]].values

y_score = outfit_df["compatibility_score"].values
y_compat = outfit_df["is_compatible"].values

# Split
Xo_tr, Xo_te, ys_tr, ys_te, yc_tr, yc_te = train_test_split(
    X_out, y_score, y_compat,
    test_size=0.2, random_state=42)
Xo_tr, Xo_val, ys_tr, ys_val, yc_tr, yc_val = train_test_split(
    Xo_tr, ys_tr, yc_tr,
    test_size=0.15, random_state=42)

scaler_out = StandardScaler()
Xo_tr  = scaler_out.fit_transform(Xo_tr)
Xo_val = scaler_out.transform(Xo_val)
Xo_te  = scaler_out.transform(Xo_te)

# ── Multi-output Model ──
inp = keras.Input(shape=(8,), name="outfit_features")
x   = layers.Dense(256, activation="relu")(inp)
x   = layers.BatchNormalization()(x)
x   = layers.Dropout(0.3)(x)
x   = layers.Dense(128, activation="relu")(x)
x   = layers.BatchNormalization()(x)
x   = layers.Dropout(0.25)(x)
x   = layers.Dense(64, activation="relu")(x)
x   = layers.Dense(32, activation="relu")(x)

# Output 1: Score (0-1)
score_out  = layers.Dense(1, activation="sigmoid",
                           name="score")(x)
# Output 2: Compatible ya nahi (binary)
compat_out = layers.Dense(1, activation="sigmoid",
                            name="compatible")(x)

outfit_model = keras.Model(
    inputs=inp,
    outputs={"score": score_out, "compatible": compat_out},
    name="OutfitCompatibilityScorer"
)

outfit_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss={
        "score":      "mse",
        "compatible": "binary_crossentropy"
    },
    loss_weights={"score": 1.0, "compatible": 0.8},
    metrics={
        "score":      "mae",
        "compatible": "accuracy"
    }
)

outfit_model.summary()

callbacks_out = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8,
        restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=4, verbose=1),
    keras.callbacks.ModelCheckpoint(
        "models/outfit_compatibility_model.h5",
        monitor="val_loss",
        save_best_only=True, verbose=0)
]

print("\n   🚀 Training Outfit Compatibility Model...")
history_out = outfit_model.fit(
    Xo_tr,
    {"score": ys_tr, "compatible": yc_tr},
    validation_data=(
        Xo_val,
        {"score": ys_val, "compatible": yc_val}
    ),
    epochs=60,
    batch_size=64,
    callbacks=callbacks_out,
    verbose=1
)

# Evaluate
results = outfit_model.evaluate(
    Xo_te,
    {"score": ys_te, "compatible": yc_te},
    verbose=0
)
print(f"\n   ✅ Outfit Compatibility Model:")
print(f"   Compatible Accuracy: {results[3]*100:.2f}%")

# Training plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history_out.history.get("score_mae",     []), label="Train MAE")
axes[0].plot(history_out.history.get("val_score_mae", []), label="Val MAE")
axes[0].set_title("Outfit Model — Score MAE")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(True)
axes[1].plot(history_out.history.get("compatible_accuracy",     []), label="Train Acc")
axes[1].plot(history_out.history.get("val_compatible_accuracy", []), label="Val Acc")
axes[1].set_title("Outfit Model — Compatibility Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(True)
plt.tight_layout()
plt.savefig("reports/outfit_training.png", dpi=120)
plt.close()
print("   📊 Training graph: reports/outfit_training.png")


# ══════════════════════════════════════════
# SCALER SAVE KARO
# ══════════════════════════════════════════
import pickle
with open("models/scaler_skin.pkl",  "wb") as f:
    pickle.dump(scaler_skin, f)
with open("models/scaler_outfit.pkl", "wb") as f:
    pickle.dump(scaler_out, f)

# Model info save karo
model_info = {
    "skin_tone_model": {
        "file":          "models/skin_tone_model.h5",
        "accuracy":      round(test_acc * 100, 2),
        "classes":       class_names,
        "input_features": ["r","g","b","brightness",
                           "r_g_diff","r_b_diff","g_b_diff","ita_angle"],
        "trained_on":    f"{len(X_tr)} samples",
        "architecture":  "Dense(128→64→32→6) + BatchNorm + Dropout"
    },
    "outfit_model": {
        "file":           "models/outfit_compatibility_model.h5",
        "accuracy":       round(results[3] * 100, 2),
        "output":         "score (0-1) + compatible (0/1)",
        "input_features": ["undertone","body_type","occasion",
                           "n_items","has_outerwear","has_accessory",
                           "color_contrast","pattern_mix"],
        "trained_on":     f"{len(Xo_tr)} samples",
        "architecture":   "Dense(256→128→64→32) + Multi-output"
    }
}

with open("models/model_info.json", "w") as f:
    json.dump(model_info, f, indent=2)

print("\n" + "="*55)
print("  🎉 Training Complete!")
print("="*55)
print(f"  Skin Tone Accuracy:      {test_acc*100:.1f}%")
print(f"  Outfit Compat Accuracy:  {results[3]*100:.1f}%")
print("\n  Files saved:")
print("  📁 models/skin_tone_model.h5")
print("  📁 models/outfit_compatibility_model.h5")
print("  📁 models/scaler_skin.pkl")
print("  📁 models/scaler_outfit.pkl")
print("  📊 reports/skin_tone_training.png")
print("  📊 reports/outfit_training.png")
print("  📊 reports/skin_tone_confusion_matrix.png")
print("\n  Ab main.py update karo — models use honge!")