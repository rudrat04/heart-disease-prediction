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
    

### Ranges for Values in the Dataset

1. **Age**: 20 to 66 years
   - Dataset range: 20 (minimum) to 66 (maximum)

2. **Gender**: 
   - 1 for male (assuming binary coding)

3. **Heart rate**: 64 to 94 beats per minute (bpm)
   - Dataset range: 64 (minimum) to 94 (maximum)
   - Normal resting heart rate for adults: 60 to 100 bpm

4. **Systolic blood pressure**: 98 to 160 mmHg
   - Dataset range: 98 (minimum) to 160 (maximum)
   - Normal range: less than 120 mmHg
   - Elevated range: 120-129 mmHg
   - High (hypertension stage 1): 130-139 mmHg
   - High (hypertension stage 2): 140 mmHg or higher

5. **Diastolic blood pressure**: 46 to 83 mmHg
   - Dataset range: 46 (minimum) to 83 (maximum)
   - Normal range: less than 80 mmHg
   - Elevated range: 80-89 mmHg (hypertension stage 1)
   - High: 90 mmHg or higher (hypertension stage 2)

6. **Blood sugar**: 160.0 to 300.0 mg/dL
   - Dataset range: 160.0 (minimum) to 300.0 (maximum)
   - Normal fasting blood sugar: 70 to 99 mg/dL
   - Prediabetes: 100 to 125 mg/dL
   - Diabetes: 126 mg/dL or higher

7. **CK-MB**: 1.08 to 13.87 ng/mL
   - Dataset range: 1.08 (minimum) to 13.87 (maximum)
   - Normal range: 0 to 3.6 ng/mL (values above may indicate heart muscle damage)

8. **Troponin**: 0.003 to 1.060 ng/mL
   - Dataset range: 0.003 (minimum) to 1.060 (maximum)
   - Normal range: 0 to 0.04 ng/mL (values above may indicate heart muscle damage)

9. **Result**: 
   - Negative: No immediate cardiac event indicated
   - Positive: Likely indicates a cardiac event, such as a heart attack

            """
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
