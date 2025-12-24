import json
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../../model/trained_logreg_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "../../model/scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def handler(event, context):
    if event["httpMethod"] != "POST":
        return {
            "statusCode": 405,
            "body": "Method Not Allowed"
        }

    data = json.loads(event["body"])

    X = np.array([[data[f"x{i}"] for i in range(1, 11)]])
    X_scaled = scaler.transform(X)

    pred = int(model.predict(X_scaled)[0])
    confidence = float(np.max(model.predict_proba(X_scaled)[0]))

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "prediction": "Risiko Tinggi" if pred == 1 else "Risiko Rendah",
            "confidence": round(confidence * 100, 2)
        })
    }
