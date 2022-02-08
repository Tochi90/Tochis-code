import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from itertools import product

st.write("""
# Car Evaluation Dataset classification app
This app classifies the **Car Evaluation dataset** specifications!
Data obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation) by Marko Bohanec.
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        buying = st.sidebar.selectbox('Buying',('low','med','high', 'vhigh'), key= "1")
        maintenance = st.sidebar.selectbox('Maintenance',('low','med','high', 'vhigh'), key= "2")
        doors = st.sidebar.selectbox('Doors', ('2', '3', '4', '5more'), key= "3")
        person = st.sidebar.selectbox('Person', ('2', '4', 'more'), key= "4")
        luggage_boot = st.sidebar.selectbox('Luggage boot', ('small','med','big'), key= "5")
        safety= st.sidebar.selectbox('Safety', ('low','med','high'), key= "6"),
        data = {'Buying': buying,
                'Maintenance': maintenance,
                'Doors': doors,
                'Person': person,
                'Luggage boot': luggage_boot,
                'Safety': safety},
        features = pd.DataFrame(
            filter(lambda x: x[0]!=x[1], product(user_input_features() .data, user_input_features() .data)),
            columns= ['Buying', 'Maintenance', 'Doors', 'Person', 'Luggage boot', 'Safety']
        )
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
cars_raw = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
                  names = ["Buying", "Maintenance", "Doors", "Person", "Luggage boot", "Safety", "Class"])
cars = cars_raw.drop(columns=['Class'])
df = pd.concat([input_df,cars],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode = ['Buying','Maintenance', 'Doors', 'Person', 'Luggage boot','Safety']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('finalized_model.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
cars_class = np.array(['good','vgood','unacc'])
st.write(cars_class[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)

