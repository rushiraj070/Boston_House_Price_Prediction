from flask import Flask, request, app,render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd


application = Flask(__name__)
app=application

scaler=pickle.load(open("Model/scaler.pkl", "rb"))
model_scaled = pickle.load(open("Model/regressor_scaled.pkl", "rb"))
model=pickle.load(open("Model/regressor.pkl", "rb"))

## Route for homepage

@app.route('/')
def index():
    return render_template('index.html')

## Route for House Price Prediction
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    result=""

    if request.method=='POST':

        CRIM=float(request.form.get("CRIM"))
        ZN = float(request.form.get('ZN'))
        INDUS = float(request.form.get('INDUS'))
        CHAS = float(request.form.get('CHAS'))
        NOX = float(request.form.get('NOX'))
        RM = float(request.form.get('RM'))
        AGE = float(request.form.get('AGE'))
        DIS = float(request.form.get('DIS'))
        RAD = int(request.form.get('RAD'))
        TAX = int(request.form.get('TAX'))
        PTRATIO = float(request.form.get('PTRATIO'))
        LSTAT = float(request.form.get('LSTAT'))

        scaled = scaler.transform([[CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,LSTAT]])
        price_min = model_scaled.predict(scaled)
        price_max = model.predict([[CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,LSTAT]])
        
         
        return render_template('house_price_prediction.html',price_min=f"${price_min[0]*1000}", price_max=f"${price_max[0]*1000}")

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")