import os
from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the model and scaler
model = pickle.load(open('diabetes_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))  # Load the saved scaler

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Hello World! Flask API is running."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'})

        # Extract values
        pre = float(data.get('Pregnancies', 0))
        glu = float(data.get('Glucose', 0))
        bp = float(data.get('BloodPressure', 0))
        st = float(data.get('SkinThickness', 0))
        ins = float(data.get('Insulin', 0))
        bmi = float(data.get('BMI', 0))
        dpf = float(data.get('DiabetesPedigreeFunction', 0))
        age = float(data.get('Age', 0))

        # Convert to numpy array and standardize it
        input_query = np.array([[pre, glu, bp, st, ins, bmi, dpf, age]])
        input_query = scaler.transform(input_query)  # Apply the same scaling

        # Make prediction
        result = model.predict(input_query)[0]

        return jsonify({'diabetes': int(result)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
