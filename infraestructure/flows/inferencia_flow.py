"""Actividad 3: pipeline de inferencia con drift emulado.

Pasos: cargar champion → ingesta de un lote de prueba con drift → predicción → guardar CSV →
registrar en MLflow.

El nivel de drift se lee de `inferencia.nivel_drift` en `configs/config.yaml`; el usuario no
lo introduce al ejecutar la pipeline. El lote se guarda en `monitoreo.lotes_dir` con una copia
histórica por ejecución, la última de cada nivel y `lote_actual.csv` para los flows de drift.

Requisitos: MLflow con un champion y, opcionalmente, Prefect:
    PREFECT_API_URL=http://localhost:4200/api
    MLFLOW_TRACKING_URI=http://localhost:5000

Ejecución (desde la raíz del proyecto):
    poetry run python -m infraestructure.flows.inferencia_flow
"""
from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

# Configurar variables de entorno por defecto si no están definidas.
if not os.environ.get("PREFECT_API_URL"):
    os.environ["PREFECT_API_URL"] = "http://localhost:4200/api"
if not os.environ.get("MLFLOW_TRACKING_URI"):
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

# Raíz del proyecto en sys.path.
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mlflow
import yaml
from mlflow.tracking import MlflowClient
from prefect import flow, task

from domain.inferencia import guardar_lote, preparar_lote, puntuar
from scripts.load_data import RUTA_CONFIG, cargar_datos, resolver_ruta


@task
def cargar_champion(cfg):
    """Carga el modelo con el alias champion. Devuelve el modelo, su versión y su run."""
    nombre = cfg["registro"]["nombre_modelo"]
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    version = MlflowClient().get_model_version_by_alias(nombre, "champion")
    modelo = mlflow.sklearn.load_model(f"models:/{nombre}@champion")
    print(f"Modelo cargado: {nombre} versión {version.version}")
    return modelo, version.version, version.run_id


@task
def ingesta(cfg):
    """Toma un lote de la parte de prueba, aplica el drift configurado y guarda una copia."""
    df = cargar_datos(cfg["datos"]["ruta"])
    nivel = cfg["inferencia"]["nivel_drift"]
    lote = preparar_lote(
        df=df,
        division=cfg["division"],
        tamano_lote=cfg["inferencia"]["tamano_lote"],
        nivel_drift=nivel,
        semilla=cfg["division"]["random_state"],
    )

    rutas = guardar_lote(lote, resolver_ruta(cfg["monitoreo"]["lotes_dir"]), nivel)
    ruta_actual = resolver_ruta(cfg["monitoreo"]["lote_actual"])
    ruta_actual.parent.mkdir(parents=True, exist_ok=True)
    if Path(rutas["actual"]).resolve() != ruta_actual.resolve():
        lote.to_csv(ruta_actual, index=False)
        rutas["actual"] = str(ruta_actual)

    print(f"Lote con drift {nivel} guardado: {rutas['historico']} ({len(lote)} filas)")
    print(f"Lote actual para monitoreo: {rutas['actual']}")
    return lote, rutas


@task
def predecir(modelo, lote, cfg):
    """Calcula la probabilidad de recomendación y guarda el resultado en CSV."""
    resultado = puntuar(modelo, lote, cfg["metricas"]["umbral"])
    carpeta = resolver_ruta(cfg["inferencia"]["carpeta_salida"])
    carpeta.mkdir(parents=True, exist_ok=True)
    nivel = cfg["inferencia"]["nivel_drift"]
    ruta = carpeta / f"prediccion_{date.today().isoformat()}_nivel_{nivel:.1f}.csv"
    ruta_actual = carpeta / f"prediccion_{date.today().isoformat()}.csv"
    resultado.to_csv(ruta, index=False)
    resultado.to_csv(ruta_actual, index=False)
    print(f"Predicciones guardadas en {ruta}")
    return str(ruta), len(resultado), float(resultado["probabilidad_recomendacion"].mean())


@task
def registrar_inferencia(version, run_id_entrenamiento, rutas, ruta_prediccion, filas, proba_media, cfg):
    """Crea un run de inferencia con tags del modelo, el nivel de drift y las rutas del lote."""
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(cfg["mlflow"]["experimento"])
    nivel = cfg["inferencia"]["nivel_drift"]
    with mlflow.start_run(run_name=f"inferencia-{date.today().isoformat()}-drift-{nivel:.1f}"):
        mlflow.set_tag("fase", "inferencia")
        mlflow.set_tag("modelo_version", version)
        mlflow.set_tag("modelo_run_id", run_id_entrenamiento)
        mlflow.set_tag("nivel_drift", nivel)
        mlflow.set_tag("lote_historico", rutas["historico"])
        mlflow.set_tag("lote_actual", rutas["actual"])
        mlflow.log_metric("filas_predichas", filas)
        mlflow.log_metric("probabilidad_media", proba_media)
        mlflow.log_artifact(ruta_prediccion)


@flow(name="inferencia-ecommerce", log_prints=True)
def inferencia_flow():
    with open(RUTA_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    modelo, version, run_id_entrenamiento = cargar_champion(cfg)
    lote, rutas = ingesta(cfg)
    ruta_prediccion, filas, proba_media = predecir(modelo, lote, cfg)
    registrar_inferencia(version, run_id_entrenamiento, rutas, ruta_prediccion, filas, proba_media, cfg)
    return {
        "lote": rutas,
        "prediccion": ruta_prediccion,
        "filas": filas,
        "nivel_drift": cfg["inferencia"]["nivel_drift"],
    }


if __name__ == "__main__":
    inferencia_flow()
