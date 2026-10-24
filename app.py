from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np

app = Flask(__name__, static_folder='static')

# FIX: Load the .keras file instead of .h5
try:
    model = tf.keras.models.load_model('breast_cancer_model.keras')
except Exception as e:
    print("Model not found. Please run train_model.py first.")
    model = None

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
        
    try:
        data = request.json
        features = np.array([[
            float(data['radius_mean']),
            float(data['texture_mean']),
            float(data['perimeter_mean']),
            float(data['area_mean']),
            float(data['smoothness_mean'])
        ]])
        
        # Get the probability
        prediction_prob = float(model.predict(features)[0][0])
        ratio_percentage = round(prediction_prob * 100, 2)
        
        # Binary classification logic
        prediction = 'Malignant' if prediction_prob > 0.5 else 'Benign'
        
        return jsonify({
            'prediction': prediction,
            'ratio': ratio_percentage
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)