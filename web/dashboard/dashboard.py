# TO RUN : $streamlit run dashboard/dashboard.py
# Local URL: http://localhost:8501
# Network URL: http://192.168.0.50:8501
# Online URL : http://15.188.179.79

import streamlit as st
from PIL import Image
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt


def main():

    API_URL = "http://127.0.0.1:5000/api/"

    # Logo "Prêt à dépenser"
    
    image = Image.open('dashboard/logo.png')
    st.sidebar.image(image, width=280)

    st.title('Tableau de bord - "Prêt à dépenser"')


    #################################################
    # LIST OF SK_ID_CURR

    # Get list of SK_IDS (cached)
    @st.cache
    def get_sk_id_list():

        # URL of the sk_id API
        SK_IDS_API_URL = API_URL + "sk_ids/"

        # Requesting the API and saving the response
        response = requests.get(SK_IDS_API_URL)

        # Convert from JSON format to Python dict
        content = json.loads(response.content)

        # Getting the values of SK_IDS from the content
        SK_IDS = content['data']

        return SK_IDS
    
    SK_IDS = get_sk_id_list()

    ##################################################
    # Selecting applicant ID
    select_sk_id = st.sidebar.selectbox('Select SK_ID from list:', SK_IDS, key=1)
    st.write('You selected: ', select_sk_id)

    ##################################################
    # FEATURES' IMPORTANCE
    ##################################################
    st.header('GLOBAL INTERPRETATION')

    # Get features importance (surrogate model, cached)
    @st.cache
    def get_features_importance():
        # URL of the features' importance API
        FEATURES_IMP_API_URL = API_URL + "features_imp"
    
        # save the response to API request
        response = requests.get(FEATURES_IMP_API_URL)
        
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))

        # convert data to pd.Series
        features_imp = pd.Series(content['data']).rename("Features importance").sort_values(ascending=False)

        return features_imp


    if st.sidebar.checkbox('Show global interpretation'):

        # get the features' importance
        features_imp = get_features_importance()

        # initialization
        sum_fi = 0
        labels = []
        frequencies = []

        # get the labels and frequencies of 10 most important features
        for feat_name, feat_imp in features_imp[:9].iteritems():
            labels.append(feat_name)
            frequencies.append(feat_imp)
            sum_fi += feat_imp

        # complete the FI of other features
        labels.append("OTHER FEATURES…")
        frequencies.append(1 - sum_fi)

        # Set up the axe
        _, ax = plt.subplots()
        ax.axis("equal")
        ax.pie(frequencies)
        ax.set_title("Features importance")
        ax.legend(
            labels,
            loc='center left',
            bbox_to_anchor=(0.7, 0.5),
        )

        # Plot the pie-plot of features importance
        st.pyplot()


        if st.checkbox('Show details'):
            st.dataframe(features_imp)


    ##################################################
    # PERSONAL DATA
    ##################################################
    st.header('PERSONAL DATA')

    # Personal data (cached)
    @st.cache
    def get_personal_data(select_sk_id):
        # URL of the scoring API (ex: SK_ID_CURR = 100005)
        PERSONAL_DATA_API_URL = API_URL + "personal_data/?SK_ID_CURR=" + str(select_sk_id)

        # save the response to API request
        response = requests.get(PERSONAL_DATA_API_URL)

        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))

        # convert data to pd.Series
        personal_data = pd.Series(content['data']).rename("SK_ID {}".format(select_sk_id))

        return personal_data

    # Aggregations of all applicants (train set, cached)
    @st.cache
    def get_aggregate():
        # URL of the aggregations API
        AGGREGATIONS_API_URL = API_URL + "aggregations"

        # Requesting the API and save the response
        response = requests.get(AGGREGATIONS_API_URL)

        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))

        # convert data to pd.Series
        data_agg = pd.Series(content['data']["0"]).rename("Population (mean/mode)")

        return data_agg


    if st.sidebar.checkbox('Show personal data'):

        # Get personal data
        personal_data = get_personal_data(select_sk_id)

        if st.checkbox('Show population data'):
            # Get aggregated data
            data_agg = get_aggregate()
            # Concatenation of the information to display
            df_display = pd.concat([personal_data, data_agg], axis=1)

        else:
            # Display only personal_data
            df_display = personal_data
        
        st.dataframe(df_display)


    ##################################################
    # SCORING
    ##################################################
    st.header('DEFAULT PROBABILITY')

    # Get scoring (cached)
    @st.cache
    def personal_scoring(select_sk_id):
        # URL of the scoring API
        SCORING_API_URL = API_URL + "scoring/?SK_ID_CURR=" + str(select_sk_id)

        # Requesting the API and save the response
        response = requests.get(SCORING_API_URL)

        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))

        # getting the values from the content
        score = content['score']

        return score

    # Get local interpretation of the score (surrogate model, cached)
    @st.cache
    def score_explanation(select_sk_id):
        # URL of the scoring API
        SCORING_EXP_API_URL = API_URL + "local_interpretation?SK_ID_CURR=" + str(select_sk_id)

        # Requesting the API and save the response
        response = requests.get(SCORING_EXP_API_URL)

        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))

        # getting the values from the content
        prediction = content['prediction']
        bias = content['bias']
        contribs =  pd.Series(content['contribs']).rename("Feature contributions")

        return (prediction, bias, contribs)


    if st.sidebar.checkbox('Show default probability'):
        # Get score
        score = personal_scoring(select_sk_id)
        # Display score (default probability)
        st.write('Default probability:', score, '%')

        if st.checkbox('Show explanations'):
            # Get prediction, bias and features contribs from surrogate model
            (_, bias, contribs) = score_explanation(select_sk_id)
            # Display the bias of the surrogate model
            st.write("Population mean (bias):", bias*100, "%")
            # Remove the features with no contribution
            contribs = contribs[contribs!=0]
            # Sorting by descending absolute values
            contribs = contribs.reindex(contribs.abs().sort_values(ascending=False).index)

            st.dataframe(contribs)

    


    ##################################################
    # FEATURES DESCRIPTIONS
    ##################################################
    st.header("FEATURES' DESCRIPTIONS")

    # Get the list of features
    @st.cache
    def get_features_descriptions():
        # URL of the aggregations API
        FEAT_DESC_API_URL = API_URL + "features_desc"

        # Requesting the API and save the response
        response = requests.get(FEAT_DESC_API_URL)

        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))

        # convert back to pd.Series
        features_desc = pd.Series(content['data']['Description']).rename("Description")

        return features_desc
    
    features_desc = get_features_descriptions()


    if st.sidebar.checkbox('Show features descriptions'):
        # Display features' descriptions
        st.table(features_desc)
    



    ################################################


if __name__== '__main__':
    main()