import os

import mlflow
import pandas as pd
from fastapi import FastAPI

MODEL_NAME = os.getenv("MODEL_NAME", "xgb")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5050")

mlflow.set_tracking_uri(MLFLOW_URI)

MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
app = FastAPI()
model = None

@app.post("/predict")
def predict(data: dict):
    global model
    try:
        model = mlflow.pyfunc.load_model(MODEL_URI)
    except Exception as e:
        return {
            "error": "Model not available",
            "cause": str(e)
        }

    df = pd.DataFrame(data["data"])
    pred = model.predict(df)
    return {
        "pred": pred.tolist()
    }