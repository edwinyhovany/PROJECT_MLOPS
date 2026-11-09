"""Pipeline de inferencia en Prefect (Actividad 2).

- Se conecta a MLflow y carga el modelo con el alias 'champion'.
- Ingesta un lote de datos que simula clientes nuevos.
- Genera predicciones y probabilidades.
- Guarda las predicciones en un archivo CSV con la fecha de ejecución.
- Registra la ejecución en MLflow con tag fase = inferencia y metadata de versión.
"""

import os
import sys
from datetime import date
from pathlib import Path

# Raíz del proyecto en sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mlflow
import yaml
from mlflow.tracking import MlflowClient
from prefect import flow, task

from scripts.load_data import cargar_datos as load_data_fn

CONFIG_FILE = ROOT_DIR / "configs" / "config.yaml"


@task(name="Cargar Modelo Champion")
def task_cargar_champion(cfg):
    """Carga el modelo mediante el alias 'champion' en MLflow."""
    nombre_modelo = cfg["registro"]["nombre_modelo"]
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    version_info = client.get_model_version_by_alias(nombre_modelo, "champion")
    model_uri = f"models:/{nombre_modelo}@champion"
    print(f"Cargando modelo champion desde {model_uri} (Versión {version_info.version})...")

    modelo = mlflow.sklearn.load_model(model_uri)
    return modelo, version_info.version, version_info.run_id


@task(name="Ingesta Lote de Clientes")
def task_ingesta_lote(cfg):
    """Obtiene una muestra de clientes para simular el lote diario."""
    tamano_lote = cfg.get("inferencia", {}).get("tamano_lote", 100)
    print(f"Ingestando lote de {tamano_lote} clientes simulados...")
    df = load_data_fn(cfg["datos"]["ruta"])
    sample_df = df.sample(n=min(tamano_lote, len(df)), random_state=None)
    return sample_df


@task(name="Generar y Guardar Predicciones")
def task_predecir_y_guardar(modelo, lote, cfg):
    """Calcula predicciones con el modelo champion y guarda en CSV con fecha."""
    cols_modelo = list(getattr(modelo, "feature_names_in_", []))
    if not cols_modelo:
        # Fallback si el pipeline no guardó feature_names_in_
        import domain.modelo as md
        cols_modelo = md.VARIABLES

    X = lote[[c for c in cols_modelo if c in lote.columns]]
    proba = modelo.predict_proba(X)[:, 1]
    umbral = cfg["metricas"]["umbral"]
    pred = (proba >= umbral).astype(int)

    resultado = lote.copy()
    resultado["probabilidad_recomendacion"] = proba
    resultado["prediccion_recomendado"] = pred

    carpeta_salida = ROOT_DIR / cfg.get("inferencia", {}).get("carpeta_salida", "data")
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    fecha_hoy = date.today().isoformat()
    ruta_csv = carpeta_salida / f"prediccion_{fecha_hoy}.csv"

    resultado.to_csv(ruta_csv, index=False)
    print(f"Predicciones guardadas exitosamente en: {ruta_csv}")
    return str(ruta_csv), len(proba), float(proba.mean())


@task(name="Registrar Inferencia en MLflow")
def task_registrar_inferencia(cfg, version, run_id_entrenamiento, ruta_csv, total_filas, proba_media):
    """Registra un run de inferencia con tag fase = inferencia."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg["mlflow"]["experimento"])

    run_name = f"inferencia-{date.today().isoformat()}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("fase", "inferencia")
        mlflow.set_tag("modelo_version", str(version))
        mlflow.set_tag("modelo_run_id", str(run_id_entrenamiento))
        mlflow.log_metric("filas_predichas", total_filas)
        mlflow.log_metric("probabilidad_media", proba_media)
        if os.path.exists(ruta_csv):
            mlflow.log_artifact(ruta_csv)
    print("Inferencia registrada en MLflow con tag fase=inferencia.")


@flow(name="pipeline-inferencia-ecommerce", log_prints=True)
def inferencia_flow():
    """Pipeline completa de inferencia Prefect."""
    with open(CONFIG_FILE, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    modelo, version, run_id_entrenamiento = task_cargar_champion(cfg)
    lote = task_ingesta_lote(cfg)
    ruta_csv, total_filas, proba_media = task_predecir_y_guardar(modelo, lote, cfg)
    task_registrar_inferencia(cfg, version, run_id_entrenamiento, ruta_csv, total_filas, proba_media)
    return ruta_csv


if __name__ == "__main__":
    inferencia_flow()
