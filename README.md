# model
# Full Machine Learning + Flask Project for Heart Failure Prediction
# Folder Structure:
# - model.ipynb
# - model.pkl
# - app.py
# - templates/
#     - index.html (with CSS)

# -------------------- STEP 1: model.ipynb --------------------
# Save the following code in a Jupyter Notebook

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("heart_failure_clinical_records_dataset (1).csv")

# Features and label
X = df.drop("DEATH_EVENT", axis=1)
y = df["DEATH_EVENT"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
preds = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, preds)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")


# -------------------- STEP 2: app.py --------------------
# Save the following in app.py

from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        data = [
            float(request.form.get("age")),
            int(request.form.get("anaemia")),
            float(request.form.get("creatinine_phosphokinase")),
            int(request.form.get("diabetes")),
            float(request.form.get("ejection_fraction")),
            int(request.form.get("high_blood_pressure")),
            float(request.form.get("platelets")),
            float(request.form.get("serum_creatinine")),
            float(request.form.get("serum_sodium")),
            int(request.form.get("sex")),
            int(request.form.get("smoking")),
            int(request.form.get("time"))
        ]
        scaled_data = scaler.transform([data])
        prediction = model.predict(scaled_data)[0]
        result = "Patient is at risk." if prediction == 1 else "Patient is not at risk."
        return render_template("index.html", result=result)

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)


# -------------------- STEP 3: templates/index.html --------------------
# Create folder: templates/index.html

# HTML with basic CSS styling

<!DOCTYPE html>
<html>
<head>
    <title>Heart Failure Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f0f4f8;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 30px;
        }
        h1 {
            color: #2c3e50;
        }
        form {
            background: #ffffff;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            width: 350px;
        }
        input[type="text"], input[type="number"] {
            width: 100%;
            padding: 8px;
            margin: 8px 0;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        input[type="submit"] {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px;
            border-radius: 5px;
            cursor: pointer;
            width: 100%;
        }
        .result {
            margin-top: 20px;
            font-weight: bold;
            color: #e74c3c;
        }
    </style>
</head>
<body>
    <h1>Heart Failure Risk Predictor</h1>
    <form method="POST">
        <input type="number" name="age" placeholder="Age" required>
        <input type="number" name="anaemia" placeholder="Anaemia (0 or 1)" required>
        <input type="number" name="creatinine_phosphokinase" placeholder="CPK Level" required>
        <input type="number" name="diabetes" placeholder="Diabetes (0 or 1)" required>
        <input type="number" name="ejection_fraction" placeholder="Ejection Fraction" required>
        <input type="number" name="high_blood_pressure" placeholder="High BP (0 or 1)" required>
        <input type="number" name="platelets" placeholder="Platelets" required>
        <input type="number" name="serum_creatinine" placeholder="Serum Creatinine" required>
        <input type="number" name="serum_sodium" placeholder="Serum Sodium" required>
        <input type="number" name="sex" placeholder="Sex (0=F, 1=M)" required>
        <input type="number" name="smoking" placeholder="Smoking (0 or 1)" required>
        <input type="number" name="time" placeholder="Follow-up period" required>
        <input type="submit" value="Predict">
    </form>
    {% if result %}
    <div class="result">{{ result }}</div>
    {% endif %}
</body>
</html>
