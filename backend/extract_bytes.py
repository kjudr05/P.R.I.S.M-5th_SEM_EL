import joblib
import os
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORIZER_PATH = os.path.join(BASE_DIR, "..", "models", "ngram_vectorizer.pkl")

vectorizer = joblib.load(VECTORIZER_PATH)

def extract_byte_features(file_path):
    # read exe as bytes
    with open(file_path, "rb") as f:
        data = f.read()

    # convert bytes to string (same as training)
    byte_string = " ".join([str(b) for b in data])

    # vectorize -> (1, 3608) sparse
    features = vectorizer.transform([byte_string])

    # 🔥 IMPORTANT: convert to dense TensorFlow tensor
    features = tf.convert_to_tensor(features.toarray(), dtype=tf.float32)

    return features
