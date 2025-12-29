from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import os

# -------------------------------
# Flask App Initialization
# -------------------------------
app = Flask(__name__)

# -------------------------------
# Load CNN Model (Safe Relative Path)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "cnn64.h5")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

cnn = load_model(MODEL_PATH, compile=False)

# -------------------------------
# Health Check (Optional but Useful)
# -------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "model_loaded": True
    })

# -------------------------------
# Prediction API
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        if "image" not in data:
            return jsonify({
                "error": "Missing 'image' field in request",
                "status": "failed"
            }), 400

        # Convert input to numpy array
        image = np.array(data["image"], dtype=np.float32)

        # Validate shape
        if image.shape != (64, 64, 1):
            return jsonify({
                "error": f"Invalid image shape {image.shape}, expected (64, 64, 1)",
                "status": "failed"
            }), 400

        image = image.reshape(1, 64, 64, 1)

        # CNN Prediction
        score = float(cnn.predict(image)[0][0])

        return jsonify({
            "cnn_score": score,
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500


# -------------------------------
# Run Server
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
