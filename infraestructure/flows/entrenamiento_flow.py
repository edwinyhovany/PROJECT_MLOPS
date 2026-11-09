"""Pipeline de entrenamiento en Prefect (Actividad 2).

Pasos: ingesta → entrenamiento → evaluación → registro.
- Cada ejecución crea un run en MLflow con el tag fase = entrenamiento.
- Registra el alias de features (features) y el número de variables (n_features).
- Compara con el champion actual en MLflow Model Registry: solo se actualiza
  el alias champion si supera el umbral_auc y supera al champion vigente.
"""

import os
import sys
from pathlib import Path

# Garantizar que la raíz del proyecto esté en el PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mlflow
import yaml
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
from sklearn.model_selection import train_test_split

import domain.modelo as modelo_domain
from scripts.load_data import cargar_datos as load_data_fn

CONFIG_FILE = ROOT_DIR / "configs" / "config.yaml"
FEATURES_FILE = ROOT_DIR / "configs" / "features.yaml"


@task(name="Ingesta de Datos")
def task_cargar_datos(ruta_datos):
    """Carga y valida el dataset asegurando tratamiento contra nulos."""
    print(f"Cargando dataset desde {ruta_datos}...")
    df = load_data_fn(ruta_datos)
    # Binarizar el objetivo
    df[modelo_domain.OBJETIVO] = modelo_domain.binarizar_objetivo(df[modelo_domain.OBJETIVO])
    print(f"Datos cargados exitosamente: {len(df)} filas, {df.shape[1]} columnas.")
    print(f"Distribución del target: {df[modelo_domain.OBJETIVO].value_counts().to_dict()}")
    return df


@task(name="Entrenamiento y Evaluación")
def task_entrenar_candidato(df, cfg, candidato, alias_features, numericas, categoricas):
    """Entrena un estimador, calcula métricas de evaluación y registra el run en MLflow."""
    division = cfg["division"]
    variables = numericas + categoricas

    train_df, test_df = train_test_split(
        df,
        test_size=division["test_size"],
        random_state=division["random_state"],
        stratify=df[modelo_domain.OBJETIVO],
    )

    X_train, y_train = train_df[variables], train_df[modelo_domain.OBJETIVO]
    X_test, y_test = test_df[variables], test_df[modelo_domain.OBJETIVO]

    tipo = candidato["tipo"]
    parametros = candidato.get("parametros", {})

    print(f"Entrenando {tipo} ({candidato['nombre_run']}) con {len(variables)} features ({alias_features})...")
    pipeline = modelo_domain.construir_modelo(tipo, parametros, numericas, categoricas)
    pipeline.fit(X_train, y_train)

    umbral = cfg["metricas"]["umbral"]
    proba = pipeline.predict_proba(X_test)[:, 1]
    metricas = modelo_domain.calcular_metricas(y_test, proba, umbral)
    print(f"Métricas obtenidas: {metricas}")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg["mlflow"]["experimento"])

    run_name = f"{candidato['nombre_run']}-{alias_features.lower()}"
    with mlflow.start_run(run_name=run_name) as run:
        # Requisito: tag fase = entrenamiento
        mlflow.set_tag("fase", "entrenamiento")
        mlflow.set_tag("features", alias_features)
        mlflow.set_tag("n_features", str(len(variables)))

        # Registrar parámetros y métricas
        mlflow.log_params({
            "modelo": tipo,
            **parametros,
            "random_state": division["random_state"],
            "umbral": umbral,
            "features": alias_features,
            "n_features": len(variables),
        })
        mlflow.log_metrics(metricas)

        # Loguear artefacto del modelo
        signature = infer_signature(X_test, proba)
        info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=signature,
            input_example=X_test.head(3),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        run_id = run.info.run_id

    return run_id, info.model_uri, metricas, candidato["nombre_run"]


@task(name="Promover a Champion")
def task_promover_champion(model_uri, metricas, cfg, run_id):
    """Evalúa el mejor modelo contra el champion vigente y actualiza alias si lo supera."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    nombre_modelo = cfg["registro"]["nombre_modelo"]
    umbral_minimo = cfg["registro"]["umbral_auc"]
    auc_nuevo = metricas["auc"]

    print(f"Evaluando promoción: AUC={auc_nuevo:.4f} (Mínimo requerido: {umbral_minimo})")

    if auc_nuevo < umbral_minimo:
        print(f"El AUC {auc_nuevo:.4f} no alcanza el umbral mínimo {umbral_minimo}. No es champion.")
        return None

    champion_actual = None
    try:
        champion_actual = client.get_model_version_by_alias(nombre_modelo, "champion")
    except (MlflowException, Exception):
        champion_actual = None

    if champion_actual is not None:
        run_champ = client.get_run(champion_actual.run_id)
        auc_champ = run_champ.data.metrics.get("auc", 0.0)
        print(f"Champion actual v{champion_actual.version} tiene AUC={auc_champ:.4f}")
        if auc_nuevo <= auc_champ:
            print(f"Nuevo AUC {auc_nuevo:.4f} no supera al champion actual ({auc_champ:.4f}). Alias no cambia.")
            return None
        print(f"Nuevo AUC {auc_nuevo:.4f} supera al champion ({auc_champ:.4f})!")
    else:
        print(f"No existe un champion previo para el modelo '{nombre_modelo}'.")

    print(f"Registrando {model_uri} en Model Registry como '{nombre_modelo}'...")
    version = mlflow.register_model(model_uri, nombre_modelo)
    client.set_registered_model_alias(nombre_modelo, "champion", version.version)
    client.set_model_version_tag(nombre_modelo, version.version, "estado", "aprobado")
    client.set_model_version_tag(nombre_modelo, version.version, "auc", f"{auc_nuevo:.4f}")
    print(f"¡Versión {version.version} promovida con éxito al alias 'champion' (AUC: {auc_nuevo:.4f})!")
    return version.version


@flow(name="pipeline-entrenamiento-ecommerce", log_prints=True)
def entrenamiento_flow():
    """Pipeline completa de entrenamiento Prefect."""
    with open(CONFIG_FILE, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    with open(FEATURES_FILE, encoding="utf-8") as f:
        feat_cfg = yaml.safe_load(f)

    alias_features = feat_cfg.get("seleccion", "ALL")
    custom_cols = feat_cfg.get("custom", [])
    numericas, categoricas = modelo_domain.resolver_features(alias_features, custom_cols)

    # 1. Ingesta
    df = task_cargar_datos(cfg["datos"]["ruta"])

    # 2. Entrenamiento y evaluación de candidatos
    candidatos = cfg.get("candidatos", [cfg.get("modelo")])
    resultados = []
    for cand in candidatos:
        run_id, model_uri, metricas, nombre_cand = task_entrenar_candidato(
            df, cfg, cand, alias_features, numericas, categoricas
        )
        resultados.append((run_id, model_uri, metricas, nombre_cand))

    # Seleccionar el mejor candidato por AUC
    mejor_run_id, mejor_uri, mejor_metricas, mejor_nombre = max(resultados, key=lambda r: r[2]["auc"])
    print(f"\nMejor candidato: {mejor_nombre} con AUC={mejor_metricas['auc']:.4f}")

    # 3. Registro / Promoción a Champion
    version_promovida = task_promover_champion(mejor_uri, mejor_metricas, cfg, mejor_run_id)

    # 4. Artefacto de resumen en Prefect
    create_markdown_artifact(
        key="resumen-entrenamiento",
        markdown=(
            f"## Resumen de Pipeline de Entrenamiento\n\n"
            f"- **Mejor Modelo**: `{mejor_nombre}`\n"
            f"- **AUC**: **{mejor_metricas['auc']:.4f}**\n"
            f"- **Accuracy**: {mejor_metricas['accuracy']:.4f}\n"
            f"- **Recall**: {mejor_metricas['recall']:.4f}\n"
            f"- **F1-Score**: {mejor_metricas['f1']:.4f}\n"
            f"- **Features**: `{alias_features}` ({len(numericas)+len(categoricas)} variables)\n"
            f"- **Run ID**: `{mejor_run_id}`\n"
            f"- **Champion Status**: {'Promovido a v' + str(version_promovida) if version_promovida else 'No promovido (sin mejora o bajo umbral)'}\n"
        ),
    )

    return mejor_run_id


if __name__ == "__main__":
    entrenamiento_flow()
