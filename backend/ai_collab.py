from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Dummy user preference vectors
USER_DATA = {
    "user1": [1, 0, 1, 0],
    "user2": [1, 1, 0, 0],
    "user3": [0, 1, 1, 1],
}

def recommend_from_users(current_user_vector):
    users = list(USER_DATA.values())

    similarities = cosine_similarity([current_user_vector], users)[0]

    best_idx = int(np.argmax(similarities))

    return f"user{best_idx+1}"