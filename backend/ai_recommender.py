import json
import numpy as np
import tensorflow as tf
import pickle
import os
import json

HISTORY_FILE = "data/history.json"

def load_history():
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except:
        return []
    score += np.random.uniform(0, 0.05)
def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
# ===== LOAD MODEL =====
model = tf.keras.models.load_model(
    "models/outfit_model.h5",
    compile=False
)

# ===== LOAD ENCODERS =====
le_skin = pickle.load(open("models/le_skin.pkl", "rb"))
le_body = pickle.load(open("models/le_body.pkl", "rb"))
le_occ  = pickle.load(open("models/le_occ.pkl", "rb"))

# ===== LOAD WARDROBE =====
def load_wardrobe():
    with open("data/wardrobe.json", "r") as f:
        return json.load(f)

# ===== FEATURE ENCODING =====
def encode_features(item, skin, body, occasion):
    skin_val = le_skin.transform([skin])[0]
    body_val = le_body.transform([body])[0]
    occ_val  = le_occ.transform([occasion])[0]

    return [
        skin_val,
        body_val,
        occ_val,
        len(item.get("colors", [])),
        len(item.get("tags", []))
    ]

# ===== PREDICT SCORE =====
def predict_score(features):
    X = np.array([features])
    score = model.predict(X, verbose=0)[0][0]
    return float(score)

# ===== MAIN OOTD FUNCTION =====
def get_ootd(skin, body, occasion):
    wardrobe = load_wardrobe()
    history = load_history()

    recent_items = [item["name"] for item in history[-5:]]  # last 5 outfits

    best_item = None
    best_score = -1

    for item in wardrobe:
        if item["name"] in recent_items:
            continue  # skip repeated outfits

        features = encode_features(item, skin, body, occasion)
        score = predict_score(features)

        if score > best_score:
            best_score = score
            best_item = item

    # fallback agar sab repeat ho gaye
    if best_item is None:
        best_item = wardrobe[0]
        best_score = 0.5

    # save to history
    history.append(best_item)
    save_history(history)

    return {
        "selected_outfit": best_item,
        "confidence": round(best_score, 2),
        "model": "AI Stylist 🔥",
        "note": "No repeat logic applied"
    }