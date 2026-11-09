"""Actividad 2: pipeline de inferencia (se ejecuta cada día, ver infraestructure/flows/desplegar.py).

Pasos: cargar modelo champion → ingesta del lote → predicción → guardar CSV → registrar en MLflow

El modelo se busca por el alias `champion`, así que cuando el pipeline de entrenamiento
asigna un nuevo champion, la inferencia usa ese modelo sin cambiar código.
El run de inferencia guarda la versión del modelo usada en sus tags, para poder conectarlo
con el entrenamiento.

Requisitos: servicios levantados, un champion registrado (ver infraestructure/flows/entrenamiento_flow.py)
y las variables:
    PREFECT_API_URL=http://localhost:4200/api
    MLFLOW_TRACKING_URI=http://localhost:5000

Ejecución de prueba:
    poetry run python infraestructure/flows/inferencia_flow.py
"""

import os
import sys
from datetime import date
from pathlib import Path

# Configurar variables de entorno por defecto si no están definidas
if not os.environ.get("PREFECT_API_URL"):
    os.environ["PREFECT_API_URL"] = "http://localhost:4200/api"
if not os.environ.get("MLFLOW_TRACKING_URI"):
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

# Raíz del proyecto en sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mlflow
import yaml
from mlflow.tracking import MlflowClient
from prefect import flow, task

from scripts.load_data import cargar_datos as leer_datos

CONFIG = str(ROOT_DIR / "configs" / "config.yaml")


@task
def cargar_champion(cfg):
    """Carga el modelo con el alias champion. Devuelve el modelo, su número de versión y su run de entrenamiento."""
    print("Buscando modelo champion en MLflow...")
    nombre = cfg["registro"]["nombre_modelo"]
    version = MlflowClient().get_model_version_by_alias(nombre, "champion")
    modelo = mlflow.sklearn.load_model(f"models:/{nombre}@champion")
    print(f"Modelo cargado: {nombre} versión {version.version}")
    return modelo, version.version, version.run_id


@task
def ingesta(cfg):
    """Toma un lote de clientes. Aquí se simula con una muestra aleatoria del CSV."""
    print(f"Ingestando lote de {cfg['inferencia']['tamano_lote']} clientes...")
    df = leer_datos(cfg["datos"]["ruta"])
    return df.sample(n=cfg["inferencia"]["tamano_lote"])


@task
def predecir(modelo, lote, cfg):
    """Calcula la probabilidad de recomendación y guarda el resultado en un CSV con la fecha de hoy."""
    print(f"Prediciendo con modelo versión {cfg['registro']['nombre_modelo']}...")
    cols_modelo = list(getattr(modelo, "feature_names_in_", []))
    if not cols_modelo:
        import domain.modelo as md
        cols_modelo = md.VARIABLES

    proba = modelo.predict_proba(lote[cols_modelo])[:, 1]
    resultado = lote.copy()
    resultado["probabilidad_recomendacion"] = proba
    resultado["prediccion"] = (proba >= cfg["metricas"]["umbral"]).astype(int)

    print("Guardando predicciones en CSV...")
    carpeta = ROOT_DIR / cfg["inferencia"]["carpeta_salida"]
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"prediccion_{date.today().isoformat()}.csv"
    resultado.to_csv(ruta, index=False)
    print(f"Predicciones guardadas en {ruta}")
    return str(ruta), len(proba), float(proba.mean())


@task
def registrar_inferencia(version, run_id_entrenamiento, ruta, filas, proba_media):
    """Crea un run de inferencia con tags que apuntan a la versión del modelo y a su run de entrenamiento."""
    print(f"Registrando inferencia para modelo versión {version}...")
    with mlflow.start_run(run_name=f"inferencia-{date.today().isoformat()}"):
        mlflow.set_tag("fase", "inferencia")
        mlflow.set_tag("modelo_version", version)
        mlflow.set_tag("modelo_run_id", run_id_entrenamiento)
        mlflow.log_metric("filas_predichas", filas)
        mlflow.log_metric("probabilidad_media", proba_media)
        if os.path.exists(ruta):
            mlflow.log_artifact(ruta)


@flow(name="inferencia-ecommerce", log_prints=True)
def inferencia_flow():
    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(cfg["mlflow"]["experimento"])
    modelo, version, run_id_entrenamiento = cargar_champion(cfg)
    lote = ingesta(cfg)
    ruta, filas, proba_media = predecir(modelo, lote, cfg)
    registrar_inferencia(version, run_id_entrenamiento, ruta, filas, proba_media)


if __name__ == "__main__":
    inferencia_flow()
