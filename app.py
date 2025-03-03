
from flask import Flask, request, jsonify
import pickle
import numpy as np

model =pickle.load(open('diabetes_model.sav','rb'))
app = Flask(__name__)

@app.route('/predict', methods=['POST'])

def predict():
    pre= request.form.get('Pregnancies')
    glu=request.form.get('Glucose')
    bp=request.form.get('BloodPressure')
    st=request.form.get('SkinThickness')
    ins=request.form.get('Insulin')
    bmi=request.form.get('BMI')
    dpf=request.form.get('DiabetesPedigreeFunction')
    age=request.form.get('Age')
    input_query= np.array([[pre,glu,bp,st,ins,bmi,dpf,age]])


    result=model.predict(input_query)[0]
    return jsonify({'diabetes':str(result)})











if __name__ == '__main__':
    app.run(debug=True)
