from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

cnn = load_model(
    r"C:\Users\kavya\Downloads\models\Core\cnn64.h5",
    compile=False
)

@app.route("/predict", methods=["POST"])
def predict():
    image = np.array(request.json["image"], dtype=np.float32)
    image = image.reshape(1, 64, 64, 1)

    score = cnn.predict(image)[0][0]
    return jsonify({"cnn_score": float(score)})

if __name__ == "__main__":
    app.run(port=5000)
