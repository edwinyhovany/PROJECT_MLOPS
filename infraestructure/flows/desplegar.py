"""Despliegues en Prefect para las pipelines de entrenamiento e inferencia.

- El entrenamiento puede lanzarse a mano.
- La inferencia tiene un horario diario.
"""

import os
from prefect.deployments import deploy
from prefect.flow_runners import SubprocessRunner
from infraestructure.flows.entrenamiento_flow import entrenar
from infraestructure.flows.inferencia_flow import inferencia


# Despliegue de entrenamiento (ejecutable a mano)
entrenamiento_deploy = deploy(
    flow=entrenar,
    name="entrenamiento-churn",
    work_pool_name="default",
    description="Pipeline de entrenamiento de modelo de churn con registro en MLflow",
)

# Despliegue de inferencia con horario diario (cada 24 horas)
inferencia_deploy = deploy(
    flow=inferencia,
    name="inferencia-champion-diario",
    work_pool_name="default",
    tags=["diario", "inferencia"],
    description="Pipeline de inferencia diaria que usa el modelo champion",
)