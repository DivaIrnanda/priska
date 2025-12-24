from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model dan scaler
model = joblib.load("model/trained_logreg_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    X = np.array([[
        data["x1"], data["x2"], data["x3"], data["x4"], data["x5"],
        data["x6"], data["x7"], data["x8"], data["x9"], data["x10"]
    ]])

    # Scaling
    X_scaled = scaler.transform(X)

    # Prediksi
    pred = int(model.predict(X_scaled)[0])
    proba = model.predict_proba(X_scaled)[0]
    confidence = float(np.max(proba))

    label = (
        "Risiko Tinggi"
        if pred == 1
        else "Risiko Rendah"
    )

    return jsonify({
        "prediction": label,
        "confidence": round(confidence * 100, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
