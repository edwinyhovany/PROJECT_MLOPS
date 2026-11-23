"""Actividad 3: despliegue de las pipelines en Prefect.

- entrenamiento-ecommerce: despliegue manual (se lanza desde la UI de Prefect).
- inferencia-ecommerce-diaria: todos los días a las 06:00 UTC.
- drift-features-ecommerce: todos los días a las 06:30 UTC, después de la inferencia.
- drift-evento-ecommerce: todos los días a las 07:00 UTC, cuando ya pueden estar las etiquetas.

Ejecución (desde la raíz del proyecto):
    poetry run python -m infraestructure.flows.desplegar

Deja esta terminal abierta: mientras esté abierta, Prefect puede lanzar las ejecuciones.
"""
import os
import sys
from pathlib import Path

# Configurar variables de entorno por defecto si no están definidas.
if not os.environ.get("PREFECT_API_URL"):
    os.environ["PREFECT_API_URL"] = "http://localhost:4200/api"

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from prefect import serve

from infraestructure.flows.drift_evento_flow import drift_evento_flow
from infraestructure.flows.drift_features_flow import drift_features_flow
from infraestructure.flows.entrenamiento_flow import entrenamiento_flow
from infraestructure.flows.inferencia_flow import inferencia_flow

if __name__ == "__main__":
    despliegues = [
        entrenamiento_flow.to_deployment(name="entrenamiento-ecommerce"),
        inferencia_flow.to_deployment(name="inferencia-ecommerce-diaria", cron="0 6 * * *"),
        drift_features_flow.to_deployment(name="drift-features-ecommerce", cron="30 6 * * *"),
        drift_evento_flow.to_deployment(name="drift-evento-ecommerce", cron="0 7 * * *"),
    ]
    print("Iniciando servicio de despliegues de Prefect...")
    for despliegue in despliegues:
        print(f"- {despliegue.name}")
    serve(*despliegues)
