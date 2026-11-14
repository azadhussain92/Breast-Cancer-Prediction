import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np

app = Flask(__name__, static_folder='static')

try:
    model = tf.keras.models.load_model('breast_cancer_model.keras')
except Exception as e:
    print(f"Model load error: {e}")
    model = None

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
        
    try:
        data = request.get_json()
        features = np.array([[
            float(data['radius_mean']),
            float(data['texture_mean']),
            float(data['perimeter_mean']),
            float(data['area_mean']),
            float(data['smoothness_mean'])
        ]], dtype=np.float32)
        
        # Direct forward pass avoids model.predict() memory spikes and worker timeouts
        prediction_prob = float(model(features, training=False).numpy()[0][0])
        ratio_percentage = round(prediction_prob * 100, 2)
        
        prediction = 'Malignant' if prediction_prob > 0.5 else 'Benign'
        
        return jsonify({
            'prediction': prediction,
            'ratio': ratio_percentage
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)