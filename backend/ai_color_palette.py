import cv2
import numpy as np
from sklearn.cluster import KMeans

def extract_colors(image_bytes, k=3):
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return []

    img = cv2.resize(img, (200, 200))
    img = img.reshape((-1, 3))

    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(img)

    colors = kmeans.cluster_centers_.astype(int)

    return colors.tolist()