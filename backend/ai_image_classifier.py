import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.applications.MobileNetV2(weights="imagenet")

FASHION_MAP = {
    "jersey": "tshirt",
    "jean": "jeans",
    "suit": "formal",
    "coat": "jacket",
    "shoe": "shoes",
    "sandal": "shoes"
}

def classify_clothing(image_bytes):
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return "unknown"

    img = cv2.resize(img, (224, 224))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)

    label = decoded[0][0][1]

    for key in FASHION_MAP:
        if key in label:
            return FASHION_MAP[key]

    return "unknown"