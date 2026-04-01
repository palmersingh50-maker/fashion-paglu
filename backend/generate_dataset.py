"""
Fashion Paglu — Dataset Generator
Dono datasets generate karta hai:
1. Skin Tone Dataset (10,000 samples)
2. Outfit Compatibility Dataset (15,000 samples)
"""

import numpy as np
import pandas as pd
import os

os.makedirs("datasets", exist_ok=True)

print("=" * 50)
print("  Fashion Paglu — Dataset Generator")
print("=" * 50)

# ══════════════════════════════════════════
# DATASET 1: SKIN TONE DATASET
# ══════════════════════════════════════════
print("\n📊 Generating Skin Tone Dataset...")

np.random.seed(42)
N_SKIN = 10000

# Skin tone classes define karo
skin_classes = {
    0: {
        "name":       "light",
        "label":      "Fair/Light",
        "r_range":    (210, 255),
        "g_range":    (180, 230),
        "b_range":    (160, 210),
        "undertone":  ["cool", "neutral"],
    },
    1: {
        "name":       "light_medium",
        "label":      "Light Medium",
        "r_range":    (185, 220),
        "g_range":    (150, 190),
        "b_range":    (120, 170),
        "undertone":  ["warm", "neutral"],
    },
    2: {
        "name":       "medium",
        "label":      "Medium/Wheatish",
        "r_range":    (160, 200),
        "g_range":    (120, 165),
        "b_range":    (90,  140),
        "undertone":  ["warm", "neutral"],
    },
    3: {
        "name":       "medium_dark",
        "label":      "Medium Dark",
        "r_range":    (130, 170),
        "g_range":    (90,  135),
        "b_range":    (60,  110),
        "undertone":  ["warm"],
    },
    4: {
        "name":       "dark",
        "label":      "Dark/Deep Brown",
        "r_range":    (90,  135),
        "g_range":    (60,  100),
        "b_range":    (40,  80),
        "undertone":  ["warm", "neutral"],
    },
    5: {
        "name":       "deep",
        "label":      "Deep/Rich",
        "r_range":    (40,  95),
        "g_range":    (25,  70),
        "b_range":    (15,  60),
        "undertone":  ["cool", "neutral"],
    },
}

skin_rows = []
samples_per_class = N_SKIN // len(skin_classes)

for class_id, props in skin_classes.items():
    for _ in range(samples_per_class):
        r = np.random.randint(*props["r_range"])
        g = np.random.randint(*props["g_range"])
        b = np.random.randint(*props["b_range"])

        # Realistic noise add karo
        r = np.clip(r + np.random.randint(-8, 8), 0, 255)
        g = np.clip(g + np.random.randint(-8, 8), 0, 255)
        b = np.clip(b + np.random.randint(-8, 8), 0, 255)

        brightness  = round(r * 0.299 + g * 0.587 + b * 0.114, 2)
        r_b_diff    = r - b
        undertone   = "warm" if r_b_diff > 20 else ("cool" if b > r + 10 else "neutral")
        ita         = round(np.degrees(np.arctan(
                        (brightness - 50) / max(abs(r_b_diff), 1))), 2)

        # Best colors for this tone
        if undertone == "warm":
            best_colors = "Camel,Rust,Olive,Mustard,Terracotta"
        elif undertone == "cool":
            best_colors = "Navy,Emerald,Purple,Burgundy,Cobalt"
        else:
            best_colors = "White,Black,Navy,Blush,Teal"

        skin_rows.append({
            "r": r, "g": g, "b": b,
            "brightness":  brightness,
            "r_g_diff":    r - g,
            "r_b_diff":    r_b_diff,
            "g_b_diff":    g - b,
            "ita_angle":   ita,
            "skin_class":  class_id,
            "skin_name":   props["name"],
            "skin_label":  props["label"],
            "undertone":   undertone,
            "best_colors": best_colors,
        })

skin_df = pd.DataFrame(skin_rows)
skin_df = skin_df.sample(frac=1, random_state=42).reset_index(drop=True)
skin_df.to_csv("datasets/skin_tone_dataset.csv", index=False)

print(f"   ✅ Skin Tone Dataset: {len(skin_df)} samples")
print(f"   📁 Saved: datasets/skin_tone_dataset.csv")
print(f"   Classes: {skin_df['skin_name'].value_counts().to_dict()}")


# ══════════════════════════════════════════
# DATASET 2: OUTFIT COMPATIBILITY DATASET
# ══════════════════════════════════════════
print("\n👔 Generating Outfit Compatibility Dataset...")

N_OUTFIT = 15000

body_types  = ["hourglass", "inverted_triangle", "pear", "rectangle", "apple", "oval"]
occasions   = ["casual", "formal", "party", "ethnic", "sports"]
undertones  = ["warm", "cool", "neutral"]
categories  = ["tops", "bottoms", "outerwear", "shoes", "accessories"]

# Compatibility rules (research-based)
COMPATIBILITY_RULES = {
    ("warm",   "inverted_triangle", "casual"):  0.95,
    ("warm",   "inverted_triangle", "formal"):  0.93,
    ("warm",   "inverted_triangle", "party"):   0.91,
    ("warm",   "hourglass",         "casual"):  0.94,
    ("warm",   "hourglass",         "formal"):  0.96,
    ("warm",   "pear",              "casual"):  0.88,
    ("warm",   "rectangle",         "casual"):  0.85,
    ("cool",   "hourglass",         "formal"):  0.97,
    ("cool",   "inverted_triangle", "casual"):  0.92,
    ("cool",   "pear",              "party"):   0.89,
    ("neutral","rectangle",         "casual"):  0.90,
    ("neutral","hourglass",         "formal"):  0.95,
}

outfit_rows = []

for _ in range(N_OUTFIT):
    undertone  = np.random.choice(undertones)
    body_type  = np.random.choice(body_types)
    occasion   = np.random.choice(occasions)
    n_items    = np.random.randint(3, 6)

    # Base compatibility score nikalo
    key        = (undertone, body_type, occasion)
    base_score = COMPATIBILITY_RULES.get(key, 0.75)

    # Factors se adjust karo
    occasion_bonus = {"formal": 0.03, "party": 0.02, "casual": 0.0,
                      "ethnic": 0.04, "sports": -0.02}
    body_bonus     = {"hourglass": 0.04, "inverted_triangle": 0.03,
                      "pear": 0.0, "rectangle": -0.01,
                      "apple": -0.02, "oval": -0.01}

    score = base_score
    score += occasion_bonus.get(occasion, 0)
    score += body_bonus.get(body_type, 0)
    score += np.random.uniform(-0.08, 0.08)   # Realistic noise
    score  = round(np.clip(score, 0.40, 0.99), 4)

    # Binary compatibility label
    is_compatible = 1 if score >= 0.75 else 0

    # Outfit features
    has_outerwear  = int(np.random.random() > 0.6)
    has_accessory  = int(np.random.random() > 0.4)
    color_contrast = round(np.random.uniform(0.1, 0.9), 2)
    pattern_mix    = int(np.random.random() > 0.7)

    outfit_rows.append({
        "undertone":        undertone,
        "body_type":        body_type,
        "occasion":         occasion,
        "n_items":          n_items,
        "has_outerwear":    has_outerwear,
        "has_accessory":    has_accessory,
        "color_contrast":   color_contrast,
        "pattern_mix":      pattern_mix,
        "undertone_enc":    undertones.index(undertone),
        "body_type_enc":    body_types.index(body_type),
        "occasion_enc":     occasions.index(occasion),
        "compatibility_score": score,
        "is_compatible":    is_compatible,
    })

outfit_df = pd.DataFrame(outfit_rows)
outfit_df = outfit_df.sample(frac=1, random_state=42).reset_index(drop=True)
outfit_df.to_csv("datasets/outfit_compatibility_dataset.csv", index=False)

print(f"   ✅ Outfit Dataset: {len(outfit_df)} samples")
print(f"   📁 Saved: datasets/outfit_compatibility_dataset.csv")
print(f"   Compatible outfits: {outfit_df['is_compatible'].sum()} / {N_OUTFIT}")
print(f"   Avg score: {outfit_df['compatibility_score'].mean():.3f}")

print("\n" + "="*50)
print("  ✅ Dono Datasets Ready!")
print("  Ab train_models.py chalao")
print("="*50)