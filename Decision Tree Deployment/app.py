import numpy as np
from flask import Flask, request, render_template
import pickle

# Create Flask app
app = Flask(__name__)

# Load the trained models
with open('dt_regressor.pkl', 'rb') as file:
    regressor_model = pickle.load(file)

with open('dt_classifier.pkl', 'rb') as file:
    classifier_model = pickle.load(file)

# Load the scaler object
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = np.array(float_features).reshape(1, -1)

    # Scale the features
    scaled_features = scaler.transform(features)

    # Predict using regression model
    regression_prediction = regressor_model.predict(scaled_features)

    # Predict using classification model
    classification_prediction = classifier_model.predict(scaled_features)

    return render_template("index.html",
                           regression_prediction=regression_prediction[0],
                           classification_prediction=classification_prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
