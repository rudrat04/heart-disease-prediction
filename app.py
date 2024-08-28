from flask import Flask, request, render_template
import pandas as pd
from pycaret.classification import load_model, predict_model

app = Flask(__name__)

# Load the model
model = load_model("meds_model_v2")


# Define the route for the homepage
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get data from form
        age = int(request.form["age"])
        gender = request.form["gender"]
        heart_rate = int(request.form["heart_rate"])
        systolic_bp = int(request.form["systolic_bp"])
        diastolic_bp = int(request.form["diastolic_bp"])
        blood_sugar = int(request.form["blood_sugar"])
        ckmb = float(request.form["ckmb"])
        troponin = float(request.form["troponin"])

        # Convert gender to binary
        if gender == "Male":
            gender = 1
        else:
            gender = 0

        # Prepare the data for prediction
        new_data = [
            [
                age,
                gender,
                heart_rate,
                systolic_bp,
                diastolic_bp,
                blood_sugar,
                ckmb,
                troponin,
            ]
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
        unseen = pd.DataFrame(new_data, columns=cols)

        # Make prediction
        prediction = predict_model(model, data=unseen)
        result = prediction["prediction_label"][0]
        score = prediction["prediction_score"][0]

        if result == "negative":
            text = "Patient does not have a heart disease"
        else:
            text = "Patient has a heart disease"

        return render_template("result.html", text=text, score=score * 100)

    return render_template("index.html")


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
