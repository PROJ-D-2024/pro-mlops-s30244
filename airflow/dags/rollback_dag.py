from __future__ import annotations

import os
from datetime import datetime
from mlflow import MlflowClient
import mlflow

from airflow import DAG
from airflow.decorators import task

ACCURACY_THRESHOLD = 0.75

def model_name():
    return os.environ.get("MODEL_NAME", "xgb")

def production_alias():
    return os.environ.get("PRODUCTION_ALIAS", "champion")

def previous_alias():
    return os.environ.get("PREVIOUS_ALIAS", "previous")

def rollbacked_alias():
    return os.environ.get("ROLLBACKED_ALIAS", "rollbacked")

mlflow.set_tracking_uri(os.environ.get("MLFLOW_EXPERIMENT_URI", "http://mlflow:5050"))

with DAG(
        dag_id="mlops_model_monitoring",
        start_date=datetime(2025, 1, 1),
        schedule="@daily",
        catchup=False,
        tags=["mlops", "demo"],
) as dag:
    @task.branch
    def check_metrics():
        client = MlflowClient()

        prod = client.get_model_version_by_alias(model_name(), production_alias())

        run = client.get_run(prod.run_id)
        accuracy = run.data.metrics.get("accuracy")

        if accuracy is None or accuracy < ACCURACY_THRESHOLD:
            return "rollback"
        return "nothing"


    @task
    def rollback():
        client = MlflowClient()

        mn = model_name()

        champion = client.get_model_version_by_alias(mn, production_alias())
        previous = client.get_model_version_by_alias(mn, previous_alias())

        client.set_registered_model_alias(
            name=mn,
            alias=production_alias(),
            version=previous.version
        )

        client.set_registered_model_alias(
            name=mn,
            alias=rollbacked_alias(),
            version=champion.version
        )

    @task
    def nothing():
        pass


    decision = check_metrics()
    decision >> [rollback(), nothing()]
