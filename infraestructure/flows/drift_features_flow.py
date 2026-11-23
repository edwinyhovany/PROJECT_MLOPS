"""Actividad 3: monitoreo de drift de features.

Compara las variables de entrada del lote que puntúa la inferencia con los datos de
entrenamiento usando Evidently. Guarda un reporte HTML por lote y emite una alerta si la
fracción de variables con drift supera `monitoreo.umbral_drift`. No necesita MLflow ni
etiquetas.

La lógica de cálculo y de alerta está en `domain/monitoreo.py`; este flow solo orquesta.

Ejecución (desde la raíz del proyecto, después de la inferencia):
    poetry run python -m infraestructure.flows.drift_features_flow
    poetry run python -m infraestructure.flows.drift_features_flow --lote lotes/lote_nivel_0.0.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Configurar variables de entorno por defecto si no están definidas.
if not os.environ.get("PREFECT_API_URL"):
    os.environ["PREFECT_API_URL"] = "http://localhost:4200/api"

# Raíz del proyecto en sys.path.
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import yaml
from prefect import flow, task

from domain.modelo import VARIABLES
from domain.monitoreo import calcular_drift_features, dividir, hay_alerta_features
from scripts.load_data import RUTA_CONFIG, cargar_datos, resolver_ruta


@task
def referencia(cfg):
    """Variables de entrada de los datos de entrenamiento: la referencia del drift."""
    entrenamiento, _ = dividir(cargar_datos(cfg["datos"]["ruta"]), cfg["division"])
    return entrenamiento[VARIABLES]


@task
def calcular_drift(datos_referencia, ruta_lote, cfg):
    """Ejecuta Evidently, guarda el reporte HTML y devuelve la fracción de variables con drift."""
    ruta_lote = resolver_ruta(ruta_lote)
    lote = pd.read_csv(ruta_lote)
    carpeta = resolver_ruta(cfg["monitoreo"]["carpeta_reportes"])
    nombre = ruta_lote.stem
    ruta_html = carpeta / f"drift_features_{nombre}.html"
    ruta_json = carpeta / f"drift_features_{nombre}.json"

    fraccion, _ = calcular_drift_features(datos_referencia, lote, ruta_html)
    umbral = cfg["monitoreo"]["umbral_drift"]
    resumen = {
        "lote": str(ruta_lote),
        "n_filas": int(len(lote)),
        "fraccion_variables_con_drift": fraccion,
        "umbral": umbral,
        "alerta": hay_alerta_features(fraccion, umbral),
        "reporte": str(ruta_html),
    }
    ruta_json.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Reporte de drift de features guardado en {ruta_html}")
    print(f"Resumen: {ruta_json}")
    return fraccion, str(ruta_html)


@task
def alerta(fraccion, umbral):
    """Avisa si la fracción de variables con drift supera el umbral. Devuelve si hay alerta."""
    hay_alerta = hay_alerta_features(fraccion, umbral)
    if hay_alerta:
        print(f"ALERTA: drift de features en {fraccion:.0%} de las variables (umbral {umbral:.0%})")
    else:
        print(f"Sin alerta: drift de features en {fraccion:.0%} de las variables (umbral {umbral:.0%})")
    return hay_alerta


@flow(name="drift-features-ecommerce", log_prints=True)
def drift_features_flow(ruta_lote: str | None = None):
    with open(RUTA_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ruta_lote = ruta_lote or cfg["monitoreo"]["lote_actual"]
    fraccion, _ = calcular_drift(referencia(cfg), ruta_lote, cfg)
    return alerta(fraccion, cfg["monitoreo"]["umbral_drift"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drift de features de un lote.")
    parser.add_argument("--lote", default=None, help="ruta del lote (por defecto monitoreo.lote_actual)")
    args = parser.parse_args()
    drift_features_flow(ruta_lote=args.lote)
