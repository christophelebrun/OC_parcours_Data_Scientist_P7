# -*- coding: utf-8 -*-
# TO RUN : $python api/api_flask.py

# Load librairies
import pandas as pd
import sklearn
import joblib
from flask import Flask, jsonify, request
import json
from treeinterpreter import treeinterpreter as ti

# Load the data
#--------------
# processed data for applying the scoring model
data_processed = pd.read_csv("data/data_processed.csv", index_col='SK_ID_CURR')
# original data for displaying personal data
data_original = pd.read_csv("data/data_original.csv", index_col='SK_ID_CURR')
# label encoded original data for interpretation with surrogate model
data_original_le = pd.read_csv("data/data_original_le.csv", index_col='SK_ID_CURR')
# aggregated data of the train set for comparison to current applicant
data_agg = pd.read_csv("data/data_agg.csv", index_col=0)
# aggregated data of the train set for comparison to current applicant
features_desc = pd.read_csv("data/features_descriptions.csv", index_col=0)

# Load the models
#----------------
# Load the scoring model
scikit_version = sklearn.__version__
model = joblib.load("models/model_{version}.pkl".format(version=scikit_version))
# Load the surrogate model
surrogate_model = joblib.load("models/surrogate_model_{version}.pkl".format(version=scikit_version))


###############################################################
app = Flask(__name__)

@app.route("/")
def loaded():
    return "API, models and data loaded…"

@app.route('/api/sk_ids/')
# Test : http://127.0.0.1:5000/api/sk_ids/
def sk_ids():
    # Extract list of 'SK_ID_CURR' from the DataFrame
    sk_ids = list(data_original.index)[:50]

    # Returning the processed data
    return jsonify({
        'status': 'ok',
        'data': sk_ids
     })


@app.route('/api/scoring/')
# Test : http://127.0.0.1:5000/api/scoring?SK_ID_CURR=346770
def scoring():
    # Parsing the http request to get arguments (applicant ID)
    SK_ID_CURR = int(request.args.get('SK_ID_CURR'))

    # Getting the data for the applicant (pd.DataFrame)
    applicant_data = data_processed.loc[SK_ID_CURR:SK_ID_CURR]

    # Converting the pd.Series to dict
    applicant_score = 100*model.predict_proba(applicant_data)[0][1]

    # Returning the processed data
    return jsonify({
        'status': 'ok',
        'SK_ID_CURR': SK_ID_CURR,
        'score': applicant_score,
     })


@app.route('/api/personal_data/')
# Test : http://127.0.0.1:5000/api/personal_data?SK_ID_CURR=346770
def personal_data():
    # Parsing the http request to get arguments (applicant ID)
    SK_ID_CURR = int(request.args.get('SK_ID_CURR'))

    # Getting the personal data for the applicant (pd.Series)
    personal_data = data_original.loc[SK_ID_CURR, :]

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

    # Converting the pd.Series to JSON
    data_agg_json = json.loads(data_agg.to_json())

    # Returning the processed data
    return jsonify({
        'status': 'ok',
        'data': data_agg_json
     })

@app.route('/api/features_desc/')
# Test : http://127.0.0.1:5000/api/features_desc
def send_features_descriptions():

    # Converting the pd.Series to JSON
    features_desc_json = json.loads(features_desc.to_json())

    # Returning the processed data
    return jsonify({
        'status': 'ok',
        'data': features_desc_json
     })

@app.route('/api/features_imp/')
# Test : http://127.0.0.1:5000/api/features_imp
def send_features_importance():
    features_importance = pd.Series(surrogate_model.feature_importances_, index=data_original_le.columns)
    
    # Converting the pd.Series to JSON
    features_importance_json = json.loads(features_importance.to_json())

    # Returning the processed data
    return jsonify({
        'status': 'ok',
        'data': features_importance_json
     })

@app.route('/api/local_interpretation/')
# Test : http://127.0.0.1:5000/api/local_interpretation?SK_ID_CURR=346770
def send_local_interpretation():

    # Parsing the http request to get arguments (applicant ID)
    SK_ID_CURR = int(request.args.get('SK_ID_CURR'))

    # Getting the personal data for the applicant (pd.DataFrame)
    local_data = data_original_le.loc[SK_ID_CURR:SK_ID_CURR]

    # Computation of the prediction, bias and contribs from surrogate model
    prediction, bias, contribs = ti.predict(surrogate_model, local_data)
    
    # Creating the pd.Series of features_contribs
    features_contribs = pd.Series(contribs[0], index=data_original_le.columns)

    # Converting the pd.Series to JSON
    features_contribs_json = json.loads(features_contribs.to_json())

    # Returning the processed data
    return jsonify({
        'status': 'ok',
        'prediction': prediction[0][0],
        'bias': bias[0],
        'contribs': features_contribs_json,
     })



#################################################
if __name__ == "__main__":
    app.run(debug=False)