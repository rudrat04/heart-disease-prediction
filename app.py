import streamlit as st
import pandas as pd
from pycaret.classification import *

st.set_page_config(
    page_title=" Heart Disease Predicton App",
    initial_sidebar_state="auto",
    page_icon="🔮",
)
st.title("Heart Disease Prediction")

st.markdown(
    """     
    
            - Age: The age of the individual.
    
            - Gender: The gender of the individual (likely coded as 1 for male and 0 for female).
            
            - Heart rate: The number of heartbeats per minute.
            
            - Systolic blood pressure: The pressure in the arteries when the heart beats.
            
            - Diastolic blood pressure: The pressure in the arteries between heartbeats.
            
            - Blood sugar: The level of glucose in the blood.
            
            - CK-MB: Creatine kinase-MB, an enzyme found in the heart muscle, elevated levels can indicate heart muscle damage.
            
            - Troponin: A protein released when the heart muscle is damaged."""
)

df = pd.read_csv("meds_cleaned_v1.csv")
cols = [
    "Age",
    "Gender",
    "Heart rate",
    "Systolic bp",
    "Diastolic bp",
    "Blood sugar",
    "CK-MB",
    "Troponin",
]

model = load_model("meds_model")

age = st.number_input("Age")
gender = st.selectbox("Gender", ["Male", "Female"])
heart_rate = st.number_input("Heart Rate")
systolic_bp = st.number_input("Systolic Blood Pressure")
diastolic_bp = st.number_input("Diastolic Blood Pressure")
blood_sugar = st.number_input("Blood Sugar")
ckmb = st.number_input("CK-MB")
troponin = st.number_input("Troponin")
if gender == "Male":
    gender = 1
else:
    gender = 0
new = [age, gender, heart_rate, systolic_bp, diastolic_bp, blood_sugar, ckmb, troponin]
unseen = pd.DataFrame([new], columns=cols)
button = st.button("Predict")

if button:
    prediction = predict_model(model, data=unseen)
    result = prediction["prediction_label"][0]
    if result == "negative":
        text = "Patient does not have a heart disease"
    else:
        text = "Patient has a heart disease"
    score = prediction["prediction_score"][0]
    st.write("Result:  ", text)
    st.write("Accuracy: ", score * 100)
