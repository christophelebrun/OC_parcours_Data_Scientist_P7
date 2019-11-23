# TO RUN : $streamlit run dashboard.py
# Local URL: http://localhost:8501
# Network URL: http://192.168.0.50:8501

import streamlit as st
import numpy
import pandas as pd
import requests
import json
import matplotlib.pyplot as plt
import matplotlib
# import shap
import xgboost

matplotlib.use('Agg')

def main():

    # Logo "Prêt à dépenser"
    from PIL import Image
    image = Image.open('logo.png')
    st.sidebar.image(image, width=280)

    st.title('Tableau de bord - "Prêt à dépenser"')


    #################################################
    # LIST OF SK_ID_CURR

    # URL of the sk_id API
    SK_IDS_API_URL = "http://127.0.0.1:5000/api/sk_ids/"

    # Requesting the API and saving the response
    response = requests.get(SK_IDS_API_URL)

    # Convert from JSON format to Python dict
    content = json.loads(response.content)

    # Getting the values of SK_IDS from the content
    SK_IDS = content['data'] #.strip('[]').split(', ')

    # Convert to list of integers
    # SK_IDS = list(map(int, SK_IDS))

    ##################################################
    # Selecting applicant ID
    select_sk_id = st.sidebar.selectbox('Select SK_ID from list:', SK_IDS, key=1)
    st.write('You selected: ', select_sk_id)

    ##################################################
    # SCORING
    ##################################################
    st.header('DEFAULT PROBABILITY')

    if st.sidebar.checkbox('Show default probability'):
        # URL of the scoring API
        SCORING_API_URL = "http://127.0.0.1:5000/api/scoring/?SK_ID_CURR=" + str(select_sk_id)

        # Requesting the API and save the response
        response = requests.get(SCORING_API_URL)

        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))

        # getting the values from the content
        score = content['score']
        st.write('Default probability:', score)


    ##################################################
    # PERSONAL DATA
    ##################################################
    st.header('PERSONAL DATA')

    if st.sidebar.checkbox('Show personal data'):

        # URL of the scoring API (ex: SK_ID_CURR = 100005)
        PERSONAL_DATA_API_URL = "http://127.0.0.1:5000/api/personal_data/?SK_ID_CURR=" + str(select_sk_id)

        # save the response to API request
        response = requests.get(PERSONAL_DATA_API_URL)

        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))

        # convert data to pd.Series
        personal_data = pd.Series(content['data']).rename(select_sk_id)


        ###################################
        ### Aggregations of all applicants (train set)
        # URL of the aggregations API
        AGGREGATIONS_API_URL = "http://127.0.0.1:5000/api/aggregations"

        # Requesting the API and save the response
        response = requests.get(AGGREGATIONS_API_URL)

        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))

        # convert data to pd.Series
        data_agg = pd.Series(content['data']).rename("Train set (mean/mode)")

        ###################################
        ### Aggregations of closer applicants (train set)



        ###################################

        # Concatenation of the information to display
        df_display = pd.concat([personal_data, data_agg], axis=1)
        st.dataframe(df_display)

    #################################################
    # SHAP

    #st.header('INTERPRETATION')

    #if st.sidebar.checkbox('Show interpretation'):
    # Local interpretation
    #######################
        # train XGBoost model
    #    X,y = shap.datasets.boston()
    #    model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

        # explain the model's predictions using SHAP values
    #    explainer = shap.TreeExplainer(model)

    #    shap_values = explainer.shap_values(X)

        # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
    #    shap.force_plot(
    #        explainer.expected_value,
    #        shap_values[0,:],
    #        X.iloc[0,:], # Selecting the local element to explain
    #        matplotlib=True,
    #        show=False)
        
    #    st.pyplot(bbox_inches='tight')
    #    plt.clf()
    
    # Global interpretation
    #######################


    ################################################


if __name__== '__main__':
   main()