from __future__ import annotations

import os
import json
from datetime import datetime

import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import mlflow
from xgboost import XGBClassifier

from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowSkipException


# -----------------------------
# Config (edit paths as you like)
# -----------------------------

def airflow_home():
    return os.environ.get("AIRFLOW_HOME", "/opt/airflow")


def base_dir():
    return os.path.join(airflow_home(), os.environ.get("PROJECT_NAME", "mlops_demo"))


def data_dir():
    return os.path.join(str(base_dir()), os.environ.get("DATA_SUBDIR", "data"))


def artifacts_dir():
    return os.path.join(str(base_dir()), os.environ.get("ARTIFACTS_SUBDIR", "artifacts"))


# MOVE THESE
# Local MLflow file store (no server required)
def mlflow_uri():
    return f"file://{os.path.join(str(base_dir()), os.environ.get("MLRUNS_SUBDIR", "mlruns"))}"


mlflow.set_tracking_uri(os.environ.get("MLFLOW_EXPERIMENT_URI", "http://mlflow:5050"))
mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "airflow-mlops-demo"))

os.makedirs(base_dir(), exist_ok=True)


# Simple dataset: UCI Heart Disease (cleaned subset via a raw GitHub URL)
def data_source():
    return os.environ.get("DATA_SOURCE",
                          "https://raw.githubusercontent.com/nmiuddin/UCI-Heart-Disease-Dataset/refs/heads/master/data/heart-disease-UCI.csv")


# Promotion rule
ACCURACY_THRESHOLD = 0.80

# KS test
DRIFT_THRESHOLD = 0.05

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 0,
}

with DAG(
        dag_id="mlops_end_to_end_demo",
        description="Ingest -> Validate -> Train -> Evaluate -> Promote (MLOps demo with MLflow logging)",
        start_date=datetime(2025, 1, 1),
        schedule="@daily",
        catchup=False,
        default_args=default_args,
        tags=["mlops", "demo"],
) as dag:
    @task
    def ingest_data() -> str:
        """Download CSV and store locally; return local path."""
        dd = data_dir()
        os.makedirs(dd, exist_ok=True)
        local_path = os.path.join(dd, "heart.csv")
        df = pd.read_csv(data_source())
        df.to_csv(local_path, index=False)
        return local_path


    @task
    def validate_data(csv_path: str) -> str:
        """Basic schema + NA checks; store a validation report."""
        ad = artifacts_dir()
        os.makedirs(ad, exist_ok=True)
        df = pd.read_csv(csv_path)

        # Minimal expectations for demo purposes
        required_cols = {"age", "sex", "cp", "trestbps", "chol", "thalach", "target"}
        missing = required_cols - set(df.columns)
        report = {
            "rows": len(df),
            "cols": list(df.columns),
            "missing_required": list(missing),
            "null_counts": df.isnull().sum().to_dict(),
        }

        report_path = os.path.join(ad, "validation_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Quick sanity checks
        if df.isnull().sum().sum() > 0:
            raise ValueError("Dataset contains null values, please clean first.")

        return report_path


    @task
    def split_data(csv_path: str) -> dict:
        """Split into train/test and write to disk; return file paths."""
        ad = artifacts_dir()
        os.makedirs(ad, exist_ok=True)
        df = pd.read_csv(csv_path)

        # Simple feature/label selection for demo
        X = df.drop(columns=["target"])
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        paths = {
            "X_train": os.path.join(ad, "X_train.parquet"),
            "X_test": os.path.join(ad, "X_test.parquet"),
            "y_train": os.path.join(ad, "y_train.parquet"),
            "y_test": os.path.join(ad, "y_test.parquet"),
        }
        X_train.to_parquet(paths["X_train"])
        X_test.to_parquet(paths["X_test"])
        pd.DataFrame({"target": y_train}).to_parquet(paths["y_train"])
        pd.DataFrame({"target": y_test}).to_parquet(paths["y_test"])

        return paths


    @task
    def train_model(paths: dict, **context) -> dict:
        """Train a logistic regression; log to MLflow; return model path."""
        ad = artifacts_dir()
        os.makedirs(ad, exist_ok=True)
        X_train = pd.read_parquet(paths["X_train"])
        y_train = pd.read_parquet(paths["y_train"])["target"]

        signature = infer_signature(X_train, y_train)
        input_example = X_train.iloc[:1]

        with mlflow.start_run(run_name="mlops_pipeline") as run:
            mlflow.set_tag("pipeline_run_id", context["run_id"])
            params = {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1, "subsample": 0.8, "random_state": 42,
                      "eval_metric": "logloss"}
            mlflow.log_params(params)

            model = XGBClassifier(**params)
            model.fit(X_train, y_train)

            model_name = "xgb"

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="models",
                registered_model_name=model_name,
                signature=signature,
                input_example=input_example
            )

            client = mlflow.tracking.MlflowClient()
            client.update_registered_model(model_name, "XGBoost classifier trained on UCI Heart Disease dataset.")

            feature_importance = pd.DataFrame({
                "feature": X_train.columns,
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)
            feature_importance_path = os.path.join(ad, "feature_importance.csv")
            feature_importance.to_csv(feature_importance_path, index=False)

            mlflow.log_artifact(feature_importance_path, artifact_path="feature_importance")

            return {
                "run_id": run.info.run_id,
                "model_name": model_name
            }


    @task
    def ks_drift(paths: dict, run_meta: dict, **context) -> dict:
        ad = artifacts_dir()
        os.makedirs(ad, exist_ok=True)

        X_train = pd.read_parquet(paths["X_train"])
        X_test = pd.read_parquet(paths["X_test"])

        drift_report = {}
        drift = False

        for column in X_train.select_dtypes(include="number"):
            stat, p_value = ks_2samp(
                X_train[column].dropna(),
                X_test[column].dropna()
            )

            drift_report[column] = {
                "ks_statistic": float(stat),
                "p_value": float(p_value),
                "drift": int(p_value < DRIFT_THRESHOLD)
            }

            if p_value < DRIFT_THRESHOLD:
                drift = True

        report_path = os.path.join(ad, "drift_report.json")
        with open(report_path, "w") as f:
            json.dump(drift_report, f, indent=2)
        with mlflow.start_run(run_id=run_meta["run_id"]):
            # mlflow.set_tag("pipeline_run_id", context["run_id"])
            mlflow.log_artifact(report_path, artifact_path="drift_report")
            mlflow.log_metric("drift", int(drift))

        return {
            "drift": drift,
            "report": drift_report
        }


    @task
    def evaluate_model(paths: dict, run_meta: dict, **context) -> dict:
        """Evaluate on test; log metrics to MLflow; return metrics dict."""
        X_test = pd.read_parquet(paths["X_test"])
        y_test = pd.read_parquet(paths["y_test"])["target"]

        model_uri = f"models:/{run_meta["model_name"]}/None"
        model = mlflow.sklearn.load_model(model_uri)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        with mlflow.start_run(run_id=run_meta["run_id"]):
            # mlflow.set_tag("pipeline_run_id", context["run_id"])
            mlflow.log_metric("accuracy", float(acc))

        metrics = {"accuracy": float(acc), "threshold": ACCURACY_THRESHOLD}
        return metrics


    @task
    def promote_if_good(metrics: dict, drift: dict, run_meta: dict) -> str:
        acc = metrics["accuracy"]

        if drift["drift"]:
            raise AirflowSkipException("Data drift detected. Not promoting.")

        if acc < ACCURACY_THRESHOLD:
            # Skipping is nice to visualize in the UI as a yellow task
            raise AirflowSkipException(
                f"Accuracy {acc:.3f} < threshold {ACCURACY_THRESHOLD:.3f}. Not promoting."
            )

        client = MlflowClient()
        model_name = run_meta["model_name"]

        version = client.get_latest_versions(model_name, stages=["None"])[0]

        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Staging",
            archive_existing_versions=True
        )


    csv_path = ingest_data()
    _report = validate_data(csv_path)
    splits = split_data(csv_path)
    run_meta = train_model(splits)
    drift = ks_drift(splits, run_meta)
    metrics = evaluate_model(splits, run_meta)
    promote_if_good(metrics, drift, run_meta)