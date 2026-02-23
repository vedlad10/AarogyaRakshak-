from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model("model/lung_cancer_iqothnccd.h5", compile=False)

classes = ["Normal", "Pneumonia", "COVID-19"]  # change if needed

def preprocess_image(image):
    image = image.resize((224, 224))  # match training size
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    image = Image.open(file.stream).convert("RGB")

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)

    predicted_class = classes[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100

    return jsonify({
        "prediction": predicted_class,
        "confidence": round(confidence, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)

