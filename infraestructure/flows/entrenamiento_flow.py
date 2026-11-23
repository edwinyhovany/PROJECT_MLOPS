"""Actividad 3: pipeline de entrenamiento (ingesta → entrenamiento → evaluación → registro).

La lógica pesada está en `domain/entrenamiento.py` y `domain/modelo.py`; este archivo solo
orquesta con Prefect y registra en MLflow. Así los tests pueden validar el entrenamiento y
la decisión de promoción sin levantar servicios.

Cada candidato de `candidatos` se entrena con cada conjunto de `features`. El mejor AUC se
compara con `registro.umbral_auc` y con el champion vigente. Solo si mejora recibe el alias
`champion` en el Model Registry.

Requisitos: MLflow y Prefect levantados (ver services/) y, opcionalmente:
    PREFECT_API_URL=http://localhost:4200/api
    MLFLOW_TRACKING_URI=http://localhost:5000

Ejecución (desde la raíz del proyecto):
    poetry run python -m infraestructure.flows.entrenamiento_flow
"""
from __future__ import annotations

import json
import os
import sys
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
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from prefect import flow, task

from domain.entrenamiento import decidir_promocion, entrenar_y_evaluar
from scripts.load_data import RUTA_CONFIG, cargar_datos as leer_datos


@task
def ingesta(ruta):
    """Carga el CSV con `scripts/load_data.py` y valida esquema, nulos y rangos."""
    print(f"Cargando datos desde {ruta}...")
    df = leer_datos(ruta)
    print(f"Datos cargados: {len(df)} filas, {df.shape[1]} columnas")
    print(f"Tasa positiva del objetivo: {df['Recommendation_Label'].mean():.2%}")
    return df


@task
def entrenar_y_registrar(df, cfg, candidato, features):
    """Entrena un candidato, calcula métricas y registra el run en MLflow."""
    nombre_run = f"{candidato['nombre_run']}-{features['nombre']}"
    print(f"Entrenando {candidato['tipo']} con features={features['nombre']} ({nombre_run})...")
    resultado = entrenar_y_evaluar(
        df=df,
        division=cfg["division"],
        features=features,
        tipo=candidato["tipo"],
        parametros=candidato["parametros"],
        umbral=cfg["metricas"]["umbral"],
        nombre_run=nombre_run,
    )
    print(f"Métricas de {nombre_run}: {resultado.metricas}")

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(cfg["mlflow"]["experimento"])
    ejemplo = df[resultado.variables].head(3)
    firma = infer_signature(ejemplo, resultado.modelo.predict_proba(ejemplo)[:, 1])

    with mlflow.start_run(run_name=nombre_run) as run:
        mlflow.set_tag("fase", "entrenamiento")
        mlflow.set_tag("algoritmo", candidato["tipo"])
        mlflow.set_tag("features", features["nombre"])
        mlflow.log_params(
            {
                "modelo": candidato["tipo"],
                **candidato["parametros"],
                "semilla": cfg["division"]["random_state"],
                "umbral": cfg["metricas"]["umbral"],
                "features": features["nombre"],
                "n_features": len(resultado.variables),
            }
        )
        mlflow.log_metrics(resultado.metricas)
        info = mlflow.sklearn.log_model(
            resultado.modelo,
            name="model",
            signature=firma,
            input_example=ejemplo,
            # Random forest y gradient boosting guardan árboles de sklearn; son modelos
            # entrenados en este proyecto, así que se confía solo en ese tipo.
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        run_id = run.info.run_id
        print(f"Run registrado en MLflow: {run_id} | URI: {info.model_uri}")

    return {
        "run_id": run_id,
        "model_uri": info.model_uri,
        "nombre_run": nombre_run,
        "features": features["nombre"],
        "tipo": candidato["tipo"],
        "metricas": resultado.metricas,
    }


@task
def promover_si_mejora(resultado, cfg):
    """Asigna el alias champion solo si el AUC supera el mínimo y al champion actual."""
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    cliente = MlflowClient()
    nombre = cfg["registro"]["nombre_modelo"]
    auc = resultado["metricas"]["auc"]

    try:
        champion = cliente.get_model_version_by_alias(nombre, "champion")
        auc_champion = cliente.get_run(champion.run_id).data.metrics["auc"]
    except (MlflowException, Exception):
        champion = None
        auc_champion = None

    promover, motivo = decidir_promocion(auc, cfg["registro"]["umbral_auc"], auc_champion)
    print(f"{resultado['nombre_run']}: {motivo}")
    if not promover:
        return None

    version = mlflow.register_model(resultado["model_uri"], nombre)
    cliente.set_registered_model_alias(nombre, "champion", version.version)
    cliente.set_model_version_tag(nombre, version.version, "estado", "aprobado")
    print(f"Versión {version.version} registrada como champion (AUC {auc:.4f})")
    return version.version


@task
def guardar_resumen(resultados, mejor, version, cfg):
    """Guarda un JSON con los resultados para la memoria y el documento de evidencia."""
    carpeta = ROOT_DIR / "reports"
    carpeta.mkdir(parents=True, exist_ok=True)
    contenido = {
        "experimento": cfg["mlflow"]["experimento"],
        "modelo_registrado": cfg["registro"]["nombre_modelo"],
        "umbral_auc": cfg["registro"]["umbral_auc"],
        "mejor_run": mejor["nombre_run"],
        "mejor_auc": mejor["metricas"]["auc"],
        "champion_version": version,
        "resultados": resultados,
    }
    ruta = carpeta / "entrenamiento_resultados.json"
    ruta.write_text(json.dumps(contenido, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Resumen del entrenamiento guardado en {ruta}")
    return str(ruta)


@flow(name="entrenamiento-ecommerce", log_prints=True)
def entrenamiento_flow():
    with open(RUTA_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    df = ingesta(cfg["datos"]["ruta"])

    resultados = []
    print(
        f"Candidatos: {[c['nombre_run'] for c in cfg['candidatos']]} | "
        f"features: {[f['nombre'] for f in cfg['features']]}"
    )
    for candidato in cfg["candidatos"]:
        for features in cfg["features"]:
            resultados.append(entrenar_y_registrar(df, cfg, candidato, features))

    mejor = max(resultados, key=lambda r: r["metricas"]["auc"])
    print(f"Mejor candidato: {mejor['nombre_run']} (AUC {mejor['metricas']['auc']:.4f})")
    version = promover_si_mejora(mejor, cfg)
    guardar_resumen(resultados, mejor, version, cfg)
    return {
        "mejor_run": mejor["run_id"],
        "auc": mejor["metricas"]["auc"],
        "champion_version": version,
        "resultados": resultados,
    }


if __name__ == "__main__":
    entrenamiento_flow()
