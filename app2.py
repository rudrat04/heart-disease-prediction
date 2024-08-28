import pandas as pd
from pycaret.classification import *
from taipy.gui import Gui

# Load your model
model = load_model("meds_model_v2")

# Prepare your data
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

# Define the app's state variables
age = 0
gender = "Male"
heart_rate = 0
systolic_bp = 0
diastolic_bp = 0
blood_sugar = 0
ckmb = 0
troponin = 0
result_text = ""
score = 0.0


# Function to perform the prediction
def predict(state):
    gender_value = 1 if state.gender == "Male" else 0
    new = [
        state.age,
        gender_value,
        state.heart_rate,
        state.systolic_bp,
        state.diastolic_bp,
        state.blood_sugar,
        state.ckmb,
        state.troponin,
    ]

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

    unseen = pd.DataFrame([new], columns=cols)
    prediction = predict_model(model, data=unseen)
    result = prediction["prediction_label"].iloc[0]

    state.result_text = (
        "Patient does not have a heart disease"
        if result == "negative"
        else "Patient has a heart disease"
    )
    state.accuracy = prediction["prediction_score"].iloc[0] * 100

    notify(state, "success", "Prediction complete!")


markdown_content = """
# Heart Disease Prediction

### Ranges for Values in the Dataset

1. **Age**: 
   - 20 to 66 years

2. **Gender**: 
   - Male or Female

3. **Heart rate**:
   - Normal resting heart rate for adults: 60 to 100 bpm

4. **Systolic blood pressure**:
   - Normal range: less than 120 mmHg
   - Elevated range: 120-129 mmHg
   - High (hypertension stage 1): 130-139 mmHg
   - High (hypertension stage 2): 140 mmHg or higher

5. **Diastolic blood pressure**: 
   - Normal range: less than 80 mmHg
   - Elevated range: 80-89 mmHg (hypertension stage 1)
   - High: 90 mmHg or higher (hypertension stage 2)

6. **Blood sugar**:
   - Normal fasting blood sugar: 70 to 99 mg/dL
   - Prediabetes: 100 to 125 mg/dL
   - Diabetes: 126 mg/dL or higher

7. **CK-MB**: 
   - Normal range: 0 to 4.9 ng/mL (values above may indicate heart muscle damage)

8. **Troponin T**: 
   - Normal range: 0 to 0.01 ng/mL (values above may indicate heart muscle damage)
   
9. **Result**: 
   - Negative
   - Positive
   - Note*: Result may not be accurate. Please check with your health care provider.
"""

layout = """
<|layout|columns=1 1|
<|
<|{markdown_content}|markdown|>
|>

<|
# Input Parameters

Age: <|{age}|number|>

Gender: <|{gender}|selector|lov=["Male", "Female"]|>

Heart Rate: <|{heart_rate}|number|>

Systolic Blood Pressure: <|{systolic_bp}|number|>

Diastolic Blood Pressure: <|{diastolic_bp}|number|>

Blood Sugar: <|{blood_sugar}|number|>

CK-MB: <|{ckmb}|number|>

Troponin T: <|{troponin}|number|>

<|Predict|button|on_action=predict|>

# Prediction Result

**Result:** <|{result_text}|>

**Accuracy:** <|{accuracy}|number|format=%.2f|>%
|>
|>
"""

gui = Gui(page=layout)
gui.run()
