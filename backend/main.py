import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os, json, uuid, shutil
from datetime import datetime
import pickle
import numpy as np
from typing import Optional
from fastapi import FastAPI, UploadFile, File
from ai_image_classifier import classify_clothing
from ai_color_palette import extract_colors
from ai_collab import recommend_from_users
from ai_recommender import get_ootd

# ── Trained Models Load Karo ──
MODELS = {}
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
DATA_DIR = os.path.join(BASE_DIR, "data")

def load_trained_models():
    """Saved .h5 models load karo"""
    try:
        MODELS["skin"] = tf.keras.models.load_model(
    os.path.join(MODELS_DIR, "skin_tone_model.h5"),
    custom_objects={"mse": tf.keras.losses.MeanSquaredError()},
    compile=False
)

        MODELS["outfit"] = tf.keras.models.load_model(
    os.path.join(MODELS_DIR, "outfit_compatibility_model.h5"),
    custom_objects={"mse": tf.keras.losses.MeanSquaredError()},
    compile=False
)
        with open(os.path.join(MODELS_DIR, "scaler_skin.pkl"),  "rb") as f:
            MODELS["scaler_skin"]  = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "scaler_outfit.pkl"), "rb") as f:
            MODELS["scaler_outfit"] = pickle.load(f)
        print("✅ Trained models loaded successfully!")
        return True
    except Exception as e:
        print(f"⚠️ Models nahi mile — OpenCV fallback use hoga: {e}")
        return False

MODELS_LOADED = load_trained_models()
# AI Modules import karo
# NOTE: Use package-style imports so this works regardless of where uvicorn is launched from.
from ai_skin_tone import detect_skin_tone
from ai_body_type import detect_body_type
from ai_recommender import get_ootd




app = FastAPI(
    title="Fashion Paglu — AI Backend",
    description="Real AI skin tone + body type detection",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(DATA_DIR,    exist_ok=True)

WARDROBE_FILE = os.path.join(DATA_DIR, "wardrobe.json")

def load_wardrobe():
    if not os.path.exists(WARDROBE_FILE):
        return []
    with open(WARDROBE_FILE, "r") as f:
        return json.load(f)

def save_wardrobe(items):
    with open(WARDROBE_FILE, "w") as f:
        json.dump(items, f, indent=2)


# ═══ ROUTES ═══

@app.get("/")
def home():
    return {
        "message":  "Fashion Paglu AI Backend chal raha hai! ✅",
        "version":  "2.0 — Real AI Integrated",
        "features": [
            "Real Skin Tone Detection (YCrCb + HSV)",
            "Real Body Type Detection (OpenCV Contour)",
            "Smart Outfit Recommendations",
            "No-Repeat OOTD (3-day window)",
            "Virtual Wardrobe with Photos"
        ]
    }


# ─── REAL AI: Selfie Analyze ───
@app.post("/analyze-photo")
async def analyze_photo(file: UploadFile = File(...)):
    contents = await file.read()

    # OpenCV se basic features nikalo
    skin_cv = detect_skin_tone(contents)
    body_cv = detect_body_type(contents)

    # ── Trained Model use karo agar available ho ──
    if MODELS_LOADED and "skin" in MODELS:
        try:
            features = np.array([[
                skin_cv["rgb"]["r"],
                skin_cv["rgb"]["g"],
                skin_cv["rgb"]["b"],
                skin_cv["brightness"],
                skin_cv["rgb"]["r"] - skin_cv["rgb"]["g"],
                skin_cv["rgb"]["r"] - skin_cv["rgb"]["b"],
                skin_cv["rgb"]["g"] - skin_cv["rgb"]["b"],
                skin_cv["ita_angle"]
            ]])
            features_scaled = MODELS["scaler_skin"].transform(features)
            predictions     = MODELS["skin"].predict(
                                  features_scaled, verbose=0)[0]
            class_names     = ["light","light_medium","medium",
                               "medium_dark","dark","deep"]
            predicted_class = class_names[np.argmax(predictions)]
            confidence      = round(float(np.max(predictions)) * 100, 1)

            skin_cv["tone"]          = predicted_class
            skin_cv["confidence"]    = confidence
            skin_cv["model_used"]    = "TensorFlow Neural Network ✅"
            skin_cv["all_probs"]     = {
                class_names[i]: round(float(predictions[i])*100, 1)
                for i in range(len(class_names))
            }
        except Exception as e:
            skin_cv["model_used"] = f"OpenCV Fallback ({str(e)})"
    else:
        skin_cv["model_used"] = "OpenCV (Model not trained yet)"

    return {
        "status":           "AI Analysis Complete ✅",
        "model_used":       skin_cv.get("model_used"),
        "skin_tone":        skin_cv["tone"],
        "tone_label":       skin_cv.get("tone_label",""),
        "undertone":        skin_cv["undertone"],
        "hex_color":        skin_cv["hex_color"],
        "confidence":       skin_cv.get("confidence", 0),
        "all_probabilities":skin_cv.get("all_probs", {}),
        "body_type":        body_cv["type"],
        "body_label":       body_cv.get("label",""),
        "body_confidence":  body_cv.get("confidence", 0),
        "recommended_colors": skin_cv["colors"],
        "styling_tips":     body_cv["tips"],
        "avoid_tips":       body_cv["avoid"],
        "ai_engine":        "TensorFlow 2.x + OpenCV"
    }

# ─── RECOMMENDATIONS ───
@app.get("/recommendations")
def recommendations(
    skin_tone:  str = "medium",
    undertone:  str = "warm",
    body_type:  str = "inverted_triangle",
    occasion:   str = "casual"
):
    outfits = get_recommendations(skin_tone, undertone, body_type, occasion)
    return {
        "outfits": outfits,
        "total":   len(outfits),
        "filters": {
            "skin_tone": skin_tone,
            "undertone": undertone,
            "body_type": body_type,
            "occasion":  occasion
        }
    }

# ─── OOTD ───
@app.get("/ootd")
def ootd(
    skin_tone:  str = "medium",
    undertone:  str = "warm",
    body_type:  str = "inverted_triangle",
    occasion:   str = "casual",
    exclude_name: Optional[str] = None,
):
    outfit = get_ootd(
        skin_tone,
        undertone,
        body_type,
        occasion,
        exclude_name=exclude_name
    )
    return {
        "outfit": outfit,
        "filters": {
            "skin_tone": skin_tone,
            "undertone": undertone,
            "body_type": body_type,
            "occasion": occasion,
        }
    }
# ─── MARK WORN ───
@app.post("/mark-worn")
def outfit_worn(outfit_name: str = Form(...)):
    result = mark_worn(outfit_name)
    return result


# ─── WARDROBE ───
@app.get("/wardrobe")
def get_wardrobe():
    items = load_wardrobe()
    return {"items": items, "total": len(items)}


@app.post("/wardrobe/add")
async def add_wardrobe_item(
    file:     UploadFile = File(...),
    name:     str = Form(...),
    category: str = Form(...),
    occasion: str = Form(...),
    color:    str = Form(...),
    season:   str = Form(...)
):
    # Photo save karo
    ext      = file.filename.split(".")[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    filepath = os.path.join(UPLOADS_DIR, filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # AI se photo analyze karo
    with open(filepath, "rb") as f:
        contents = f.read()
    skin = detect_skin_tone(contents)

    new_item = {
        "id":        str(uuid.uuid4()),
        "name":      name,
        "category":  category,
        "occasion":  occasion,
        "color":     color,
        "season":    season,
        "photo":     filename,
        # Keep photo_url relative so it works across backend ports.
        "photo_url": f"/uploads/{filename}",
        "added_on":  datetime.now().strftime("%Y-%m-%d"),
        "worn_count": 0,
        "last_worn":  None,
        "ai_tags": {
            "dominant_color": skin["hex_color"],
            "undertone_match": skin["undertone"],
            "skin_compatible": skin["tone"]
        }
    }

    wardrobe = load_wardrobe()
    wardrobe.append(new_item)
    save_wardrobe(wardrobe)

    return {
        "message": f"{name} wardrobe mein add ho gaya! ✅",
        "item":    new_item,
        "ai_analysis": {
            "color_detected": skin["hex_color"],
            "tone_match":     skin["undertone"]
        }
    }


@app.delete("/wardrobe/{item_id}")
def delete_item(item_id: str):
    wardrobe = [i for i in load_wardrobe() if i["id"] != item_id]
    save_wardrobe(wardrobe)
    return {"message": "Item delete ho gaya! 🗑"}


app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

@app.post("/classify-clothing")
async def classify(file: UploadFile = File(...)):
    content = await file.read()
    result = classify_clothing(content)
    return {"category": result}
@app.post("/extract-colors")
async def colors(file: UploadFile = File(...)):
    content = await file.read()
    colors = extract_colors(content)
    return {"colors": colors}
@app.get("/collab")
def collab():
    user_vec = [1, 0, 1, 0]
    result = recommend_from_users(user_vec)
    return {"similar_user": result}
@app.get("/enhanced")
def enhanced():
    result = enhanced_recommendation("medium", "warm", "rectangle", "casual")
    return result