# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
import pickle

# Load the RandomForest classifier from the pickle file
with open("classifier.pkl" ,"rb") as pickle_in:
    classifier = pickle.load(pickle_in)
with open("scaler.pkl","rb") as scaler_file:
    scaler = pickle.load(scaler_file)
def predict_potability(ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity):
    # Make predictions using the loaded classifier
    standardized_input = scaler.transform([[ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]])
    prediction = classifier.predict(standardized_input)
    return prediction[0]  # Assuming the prediction is a single value

def main():
    st.title("Water Potability Prediction")

    html_temp = """
    <div style="background-color:red;padding:10px">
    <h2 style="color:black;text-align:center;">Water Potability Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Input fields for user to enter feature values
    ph = st.text_input("pH", "")
    Hardness = st.text_input("Hardness","")
    Solids = st.text_input("Solids", "")
    Chloramines = st.text_input("Chloramines", "")
    Sulfate = st.text_input("Sulfate", "")
    Conductivity = st.text_input("Conductivity", "")
    Organic_carbon = st.text_input("Organic_carbon", "")
    Trihalomethanes = st.text_input("Trihalomethanes", "")
    Turbidity = st.text_input("Turbidity", "")

    result = ""

    if st.button("Predict"):
        try:
            # Convert inputs to float and make prediction
            prediction = predict_potability(float(ph), float(Hardness), float(Solids), float(Chloramines), float(Sulfate), float(Conductivity), float(Organic_carbon), float(Trihalomethanes), float(Turbidity))
            if prediction == 1:
                st.success('The water is predicted to be potable.')
            elif prediction==0:
                st.error('The water is predicted to be non-potable.')
        except ValueError:
            # Handle potential conversion errors from text input
            st.error("Please enter valid numbers for all features.")

    if st.button("About"):
        st.text("Built with Streamlit")

if __name__ == '__main__':
        main()
