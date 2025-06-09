from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os
import logging

# Setup logging (non-debug mode untuk production)
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Load model
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")
try:
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    app.logger.info("Model loaded successfully from %s", MODEL_PATH)
except Exception as e:
    app.logger.error("Failed to load model: %s", e)
    model = None  # Prevent crashing if model failed

# Constants
FEATURE = ['suhu', 'ph_air', 'intensitas_cahaya']
LABEL = ['Bayam', 'Kangkung', 'Pakcoy', 'Sawi', 'Selada']

@app.route('/')
def index():
    return jsonify({
        "message": "Welcome to the Plant Recommendation API. Use POST /predict to get recommendation."
    })

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({
            "status": "ERROR",
            "message": "Model is not loaded."
        }), 500

    data = request.get_json()

    try:
        suhu = float(data.get('suhu', None))
        ph_air = float(data.get('ph_air', None))
        intensitas_cahaya = float(data.get('intensitas_cahaya', None))

        # Validate input
        if any(v is None for v in [suhu, ph_air, intensitas_cahaya]):
            raise ValueError("Missing one or more required fields: suhu, ph_air, intensitas_cahaya")

        new_data = pd.DataFrame([[suhu, ph_air, intensitas_cahaya]], columns=FEATURE)
        res = model.predict(new_data)
        result_label = LABEL[res[0]]

        return jsonify({
            "status": "SUCCESS",
            "result": {
                "name": result_label,
                "description": f"Tanaman {result_label} cocok untuk kondisi suhu {suhu}°C, pH {ph_air}, dan intensitas cahaya {intensitas_cahaya}.",
                "image": f"{result_label.lower()}.png"
            }
        }), 200

    except Exception as e:
        app.logger.warning("Prediction error: %s", str(e))
        return jsonify({
            "status": "ERROR",
            "message": "Input tidak valid atau gagal memproses prediksi."
        }), 400

# Dynamic port binding for Railway
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port)
