from flask import Flask,render_template,jsonify,request
app = Flask(__name__)
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler



app = Flask(__name__)

standard_scaler = pickle.load(open('models/scaler.pkl','rb'))
ridge_model = pickle.load(open('models/ridge.pkl','rb'))



@app.route('/',methods =['GET','POST'])
def predict_datapoint():
    try:


        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))

        new_data_scaled = standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridge_model.predict(new_data_scaled)
        return render_template('home.html', result='THE FWI prediction is : {}'.format(result[0]))
    except Exception as e:
        return render_template('home.html', result='Please Enter Valid Data')





if __name__== "__main__":
    app.run(host="0.0.0.0")