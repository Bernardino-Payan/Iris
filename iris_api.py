import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.datasets import load_iris

# Load trained model
model = joblib.load("iris_model.pkl")

# Load feature names
iris = load_iris()
feature_names = iris.feature_names  # Example: ['sepal length', 'sepal width', 'petal length', 'petal width']

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", feature_names=feature_names)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [float(request.form[f]) for f in feature_names]
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)[0]
        predicted_class = iris.target_names[prediction]
        return render_template("index.html", feature_names=feature_names, prediction=predicted_class)
    except Exception as e:
        return render_template("index.html", feature_names=feature_names, error=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Get PORT from Heroku, default to 5000 locally
    app.run(host="0.0.0.0", port=port)
