import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("models/body_type_model.h5")

classes = ['hourglass', 'inverted_triangle', 'pear', 'rectangle']

def detect_body_type(image_path):
    img = Image.open(image_path).resize((128,128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    idx = np.argmax(preds)

    return {
        "type": classes[idx],
        "confidence": float(np.max(preds))
    }