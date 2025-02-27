import joblib
import numpy as np
from flask import Flask, request, jsonify
from sklearn.datasets import load_iris

# Load trained model
model = joblib.load("iris_model.pkl")

# Load Iris feature names
iris = load_iris()
feature_names = iris.feature_names  # Example: ['sepal length', 'sepal width', 'petal length', 'petal width']

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "message": "Welcome to the Iris Prediction API.",
        "instructions": "Use the /features endpoint to see expected input format. Send a POST request to '/predict' with JSON payload containing 'features'.",
        "example_request": {
            "features": [5.1, 3.5, 1.4, 0.2]
        }
    })

@app.route("/features", methods=["GET"])
def get_features():
    return jsonify({
        "feature_names": feature_names,
        "example_values": [5.1, 3.5, 1.4, 0.2]
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)[0]
        predicted_class = iris.target_names[prediction]
        return jsonify({"prediction": predicted_class})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
