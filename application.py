from flask import Flask,request,render_template,jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

# import lassocv model and standarscaler pickle file
lasso_cv=pickle.load(open("models/lassocv.pkl","rb"))
scaler_pkl=pickle.load(open("models/scaler.pkl","rb"))

@app.route("/")
def Forest_page():
    return render_template("index.html")

@app.route("/predict_FWI",methods=['GET','POST'])
def Forest_fire():
    if (request.method=="POST"):

        #return render_template('index.html')
        Temperature=float(request.form['Temperature'])
        RH=float(request.form['RH'])
        WS=float(request.form['WS'])
        Rain=float(request.form['Rain'])
        FFMC=float(request.form['FFMC'])
        DMC=float(request.form['DMC'])
        ISI=float(request.form['ISI'])
        Classes=float(request.form['Classes'])
        Region=float(request.form['Region'])

        new_scaled_data=scaler_pkl.transform([[Temperature,RH,WS,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=lasso_cv.predict(new_scaled_data)

        return render_template('home.html',results=result[0])
        
    else:
        return render_template("home.html")

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8081)
