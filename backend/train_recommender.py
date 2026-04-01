import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

# Load dataset
df = pd.read_csv("dataset_outfit.csv")

# Encode categorical data
le_skin = LabelEncoder()
le_body = LabelEncoder()
le_occ  = LabelEncoder()

df["skin"] = le_skin.fit_transform(df["skin"])
df["body"] = le_body.fit_transform(df["body"])
df["occasion"] = le_occ.fit_transform(df["occasion"])

# Features + Target
X = df[["skin", "body", "occasion", "color_count", "tag_count"]]
y = df["score"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Train
model.fit(X_scaled, y, epochs=20, batch_size=8)

# Save model
model.save("models/outfit_compatibility_model.h5")

# Save scaler + encoders
pickle.dump(scaler, open("models/scaler_outfit.pkl", "wb"))
pickle.dump(le_skin, open("models/le_skin.pkl", "wb"))
pickle.dump(le_body, open("models/le_body.pkl", "wb"))
pickle.dump(le_occ, open("models/le_occ.pkl", "wb"))

print("✅ Model trained and saved!")
model.save("models/outfit_model.h5")