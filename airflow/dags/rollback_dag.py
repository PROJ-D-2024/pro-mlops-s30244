from __future__ import annotations

import os
from datetime import datetime
from mlflow import MlflowClient

from airflow import DAG
from airflow.decorators import task

ACCURACY_THRESHOLD = 0.75

def model_name():
    return os.environ.get("MODEL_NAME", "xgb")

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

        prod = client.get_model_version_by_alias(model_name(), "champion")

        run = client.get_run(prod.run_id)
        accuracy = run.data.metrics.get("accuracy")

        if accuracy is None or accuracy < ACCURACY_THRESHOLD:
            return "rollback"
        return "nothing"


    @task
    def rollback():
        client = MlflowClient()

        mn = model_name()

        champion = client.get_model_version_by_alias(mn, "champion")
        previous = client.get_model_version_by_alias(mn, "previous")

        client.set_registered_model_alias(
            name=mn,
            alias="champion",
            version=previous.version
        )

        client.set_registered_model_alias(
            name=mn,
            alias="rollbacked",
            version=champion.version
        )

    @task
    def nothing():
        pass


    decision = check_metrics()
    decision >> [rollback(), nothing()]
