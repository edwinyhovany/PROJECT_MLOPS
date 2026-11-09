"""Despliegues en Prefect (Actividad 2).

- Pipeline de entrenamiento: despliegue manual (sin horario fijo).
- Pipeline de inferencia: despliegue programado con horario diario (cron: todos los días a las 06:00 UTC).
"""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from prefect import serve

from infraestructure.flows.entrenamiento_flow import entrenamiento_flow
from infraestructure.flows.inferencia_flow import inferencia_flow

if __name__ == "__main__":
    despliegue_entrenamiento = entrenamiento_flow.to_deployment(
        name="entrenamiento-ecommerce",
        description="Pipeline de entrenamiento que entrena modelos, registra en MLflow y promueve a champion",
    )
    despliegue_inferencia = inferencia_flow.to_deployment(
        name="inferencia-ecommerce-diaria",
        cron="0 6 * * *",
        description="Pipeline diaria de inferencia que carga el modelo champion y genera predicciones",
    )

    print("Iniciando servicio de despliegues de Prefect...")
    print("- Despliegue 1: entrenamiento-ecommerce (Manual)")
    print("- Despliegue 2: inferencia-ecommerce-diaria (Diario: cron 0 6 * * *)")
    serve(despliegue_entrenamiento, despliegue_inferencia)
