import tensorflow as tf
import numpy as np
import cv2

# Load trained model
model = tf.keras.models.load_model(
    "models/skin_tone_model.h5",
    compile=False
)

# Classes (training ke time same hone chahiye)
TONE_CLASSES = [
    "light",
    "light_medium",
    "medium",
    "medium_dark",
    "dark",
    "deep"
]

UNDERTONE_CLASSES = ["warm", "cool", "neutral"]


def detect_skin_tone(image_bytes):
    """
    FULL AI Skin Tone Detection
    No rules, no thresholds — only ML prediction
    """

    # Bytes → image
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return _default_skin()

    # Resize for model
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    pred = model.predict(img, verbose=0)[0]

    # Split prediction (assume model outputs both)
    tone_idx = np.argmax(pred[:6])
    undertone_idx = np.argmax(pred[6:])

    tone = TONE_CLASSES[tone_idx]
    undertone = UNDERTONE_CLASSES[undertone_idx]

    confidence = float(np.max(pred))

    return {
        "tone": tone,
        "undertone": undertone,
        "confidence": round(confidence * 100, 2),
        "model": "deep_learning"
    }


def _default_skin():
    return {
        "tone": "medium",
        "undertone": "neutral",
        "confidence": 0,
        "model": "fallback"
    }