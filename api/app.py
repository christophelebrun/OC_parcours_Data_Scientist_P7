# -*- coding: utf-8 -*-

# TO RUN : $python app.py

# Load librairies
from flask import Flask, render_template, jsonify, request
import json
import requests
import os
import pandas as pd
import sklearn
import joblib


# Load data test from loans applications
dir_name = '../data/cleaned'
file_name = 'data_test.csv'
file_path = os.path.join(dir_name, file_name)
data_test = pd.read_csv(file_path, index_col='SK_ID_CURR')

# Load data train from loans applications
file_name = 'data_train.csv'
file_path = os.path.join(dir_name, file_name)
data_train = pd.read_csv(file_path, index_col='SK_ID_CURR')

# Load the model
scikit_version = sklearn.__version__
model = joblib.load("../models/model_{version}.pkl".format(version=scikit_version))



###############################################################
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route('/dashboard/')
def dashboard():
    return render_template("dashboard.html")

@app.route('/api/sk_ids/')
# Test : http://127.0.0.1:5000/api/sk_ids/
def sk_ids():
    # Extract list of 'SK_ID_CURR' from the DataFrame
    sk_ids = list(data_test.index)[:50]

    # Convering to JSON
    sk_ids_json = json.dumps(sk_ids)

    # Returning the processed data
    return jsonify({
        'status': 'ok',
        'data': sk_ids #sk_ids_json
     })


@app.route('/api/personal_data/')
# Test : http://127.0.0.1:5000/api/personal_data?SK_ID_CURR=100001
def personal_data():
    # Parsing the http request to get arguments (applicant ID)
    SK_ID_CURR = int(request.args.get('SK_ID_CURR'))

    # Getting the personal data for the applicant (pd.Series)
    personal_data = data_test.loc[SK_ID_CURR, :]

    # Converting the pd.Series to JSON
    personal_data_json = json.loads(personal_data.to_json())
    

    # Returning the processed data
    return jsonify({
        'status': 'ok',
        'data': personal_data_json
     })


@app.route('/api/aggregations/')
# Test : http://127.0.0.1:5000/api/aggregations
def aggregations():

    # Aggregate the data from loan applications
    data_agg_num = data_train.mean(numeric_only=True)
    data_agg_cat = data_train.select_dtypes(exclude='number').mode().iloc[0]
    data_agg = pd.concat([data_agg_num, data_agg_cat])

    # Converting the pd.DataFrame to JSON
    data_agg_json = json.loads(data_agg.to_json())
    
    # Returning the processed data
    return jsonify({
        'status': 'ok',
        'data': data_agg_json
     })

@app.route('/api/scoring/')
# Test : http://127.0.0.1:5000/api/scoring?SK_ID_CURR=100001
def scoring():
    # Parsing the http request to get arguments (applicant ID)
    SK_ID_CURR = int(request.args.get('SK_ID_CURR'))

    # Getting the data for the applicant (pd.DataFrame)
    applicant_data = data_test.loc[SK_ID_CURR:SK_ID_CURR]

    # Converting the pd.Series to dict
    applicant_score = 100*model.predict_proba(applicant_data)[0][1]

    # Returning the processed data
    return jsonify({
        'status': 'ok',
        'SK_ID_CURR': SK_ID_CURR,
        'score': applicant_score,
     })


###############################################
# Remplacez cette ligne par votre clé OPENWEATHERMAP
METEO_API_KEY = "186d9f11d721d200426dad2cd4d046b2"

if METEO_API_KEY is None:
    # URL de test :
    METEO_API_URL = "https://samples.openweathermap.org/data/2.5/forecast?lat=0&lon=0&appid=xxx"
else: 
    # URL avec clé :
    METEO_API_URL = "https://api.openweathermap.org/data/2.5/forecast?lat=48.883587&lon=2.333779&appid=" + METEO_API_KEY



@app.route('/api/meteo/')
def meteo():
    # save the response to API request
    response = requests.get(METEO_API_URL)
    # convert from JSON format to Python dict
    content = json.loads(response.content.decode('utf-8'))

    # if all didn't went well...
    if response.status_code != 200:
        return jsonify({
            'status': 'error',
            'message': 'La requête à l\'API météo n\'a pas fonctionné. Voici le message renvoyé par l\'API : {}'.format(content['message'])
            }), 500    

    # Looping over all previsions
    data = []  # On initialise une liste vide
    for prev in content["list"]:
        datetime = prev['dt'] * 1000  # Conversion to milliseconds
        temperature = prev['main']['temp'] - 273.15 # Conversion de Kelvin en °c
        temperature = round(temperature, 2)
        data.append([datetime, temperature])

    # Returning the processed data
    return jsonify({
        'status': 'ok',
        'data': data
    })



#################################################
if __name__ == "__main__":
    app.run(debug=True)