from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model & scaler
##model = pickle.load(open("student_model.pkl", "rb"))
rf = pickle.load(open("scaler.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            hours_studied = float(request.form["hours_studied"])
            previous_scores = float(request.form["previous_scores"])
            extracurricular = int(request.form["extracurricular"])
            sleep_hours = float(request.form["sleep_hours"])
            sample_papers = int(request.form["sample_papers"])

            # Create input array (same order as training)
            features = np.array([[ 
                hours_studied,
                previous_scores,
                extracurricular,
                sleep_hours,
                sample_papers
            ]])

            # Scale features
           

            # Predict
            prediction = rf.predict(features)[0]
            prediction = round(prediction, 2)

        except Exception as e:
            prediction = "Error: " + str(e)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
