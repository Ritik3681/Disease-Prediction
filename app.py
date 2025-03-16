from flask import Flask, request, jsonify
import pickle
import numpy as np

# ✅ Load the trained model and scaler
model = pickle.load(open('diabetes_model .sav', 'rb'))
scaler = pickle.load(open('scaler .sav', 'rb'))
model1 = pickle.load(open('heart_disease_model.sav', 'rb'))
scaler1 = pickle.load(open('scaler1.sav', 'rb'))

# ✅ Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to Diabetes Prediction API! 🚀"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ✅ Ensure JSON input is received
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'})

        # ✅ Extract input values
        input_data = np.array([[
            float(data.get('Pregnancies', 0)),
            float(data.get('Glucose', 0)),
            float(data.get('BloodPressure', 0)),
            float(data.get('SkinThickness', 0)),
            float(data.get('Insulin', 0)),
            float(data.get('BMI', 0)),
            float(data.get('DiabetesPedigreeFunction', 0)),
            float(data.get('Age', 0))
        ]])

        # ✅ Standardize input using the same scaler
        std_data = scaler.transform(input_data)
        print("Standardized Input:", std_data)

        # ✅ Make prediction
        result = model.predict(std_data)[0]
        return jsonify({'diabetes': int(result)})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict_heart', methods=['POST'])
def predict_heart():
    try:
        # ✅ Ensure JSON input is received
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'})

        # ✅ Extract input values
        input_data = np.array([[
            float(data.get('age', 0)),
            float(data.get('sex', 0)),
            float(data.get('cp', 0)),
            float(data.get('trestbps', 0)),
            float(data.get('chol', 0)),
            float(data.get('fbs', 0)),
            float(data.get('restecg', 0)),
            float(data.get('thalach', 0)),
            float(data.get('exang', 0)),
            float(data.get('oldpeak', 0)),
            float(data.get('slope', 0)),
            float(data.get('ca', 0)),
            float(data.get('thal', 0))
        ]])

        # ✅ Standardize input using the same scaler
        std_data = scaler1.transform(input_data)
        print("Standardized Input:", std_data)

        # ✅ Make prediction
        result = model1.predict(std_data)[0]
        return jsonify({'heart_disease': int(result)})

    except Exception as e:
        return jsonify({'error': str(e)})

# ✅ Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
