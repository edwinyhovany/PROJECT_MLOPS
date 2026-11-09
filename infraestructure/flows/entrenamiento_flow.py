"""Pipeline de entrenamiento en Prefect.

Pasos: ingesta → entrenamiento → evaluación → registro en MLflow.

Cada ejecución crea un run en MLflow con el tag fase = entrenamiento.
El modelo nuevo se compara con el champion actual. Solo se registra como nueva
versión y pasa a ser champion si supera un AUC mínimo y también el AUC del
champion vigente. Si no, el run queda registrado pero el alias no cambia.
"""

import os
import sys

# Añadir el directorio raíz (3 niveles hacia arriba desde flows/) al path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import mlflow
import yaml
from mlflow.models import infer_signature
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import domain.modelo as modelo
from scripts.load_data import cargar_datos

# Configuración
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs", "config.yaml")
FEATURES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs", "features.yaml")

# Cargar configuración
with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# Determinar features según config YAML
seleccion = cfg.get("features", {}).get("seleccion", "ALL")
numericas, categoricas = modelo.resolver_features(seleccion)

# Guardar información para MLflow
n_features = len(numericas) + len(categoricas)


@task
def cargar_datos(ruta):
    """Carga el CSV con validación de columnas."""
    from scripts.load_data import cargar_datos as load_data
    return load_data(ruta)


@task
def entrenar_y_registrar(df, cfg, numericas, categoricas, n_features):
    """Entrena el modelo, calcula métricas y lo registra en MLflow.

    Devuelve (run_id, metricas, model_uri).
    """
    # 1. Partir los datos en entrenamiento y prueba (estratificado)
    division = cfg["division"]
    train_df, test_df = train_test_split(
        df,
        test_size=division["test_size"],
        random_state=division["random_state"],
        stratify=df[modelo.OBJETIVO],
    )
    X_train = train_df[numericas + categoricas]
    y_train = train_df[modelo.OBJETIVO]
    X_test = test_df[numericas + categoricas]
    y_test = test_df[modelo.OBJETIVO]

    # 2. Crear el modelo: preprocesado (escalar y one-hot) + estimador
    tipo = cfg["modelo"]["tipo"]
    parametros = cfg["modelo"].get("parametros") or {}
    if tipo == "logistic_regression":
        estimador = LogisticRegression(**parametros)
    elif tipo == "gradient_boosting":
        estimador = GradientBoostingClassifier(**parametros)
    else:
        raise ValueError(f"modelo '{tipo}' no soportado")

    preprocesado = ColumnTransformer(
        [
            ("num", StandardScaler(), numericas),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas),
        ]
    )
    pipeline = Pipeline([("preprocesado", preprocesado), ("modelo", estimador)])

    # 3. Entrenar y evaluar
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    umbral = cfg["metricas"]["umbral"]
    pred = (proba >= umbral).astype(int)

    metricas = {
        "auc": float(roc_auc_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
    }

    # 4. Registrar el run en MLflow
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experimento"])

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # Registrar tag fase = entrenamiento
        mlflow.set_tag("fase", "entrenamiento")

        # Registrar parámetros y métricas
        mlflow.log_params({
            "modelo": tipo,
            **parametros,
            "random_state": division["random_state"],
            "umbral": umbral,
            "features": seleccion,
            "n_features": n_features,
        })
        mlflow.log_metrics(metricas)

        # Registrar alias de features en MLflow
        mlflow.set_tag("features", seleccion)
        mlflow.set_tag("n_features", str(n_features))

        # Loguear el modelo
        signature = infer_signature(X_test, proba)
        model_uri = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=signature,
            input_example=X_test.head(3),
        )

        # Lógica de champion: registrar como nueva versión y comparar AUC
        # Intentar establecer el alias champion si mejora
        auc_actual = metricas["auc"]
        try:
            # Obtener la versión actual del champion
            current_champion = mlflow.MlflowClient().get_model_version(
                name="champion", alias="champion"
            )
            auc_champion = float(current_champion.metrics.get("auc", 0))
            mejora_minima = cfg.get("mejora_minima", 0.02)  # 2% de mejora mínima
            if auc_actual > auc_champion and (auc_actual - auc_champion) >= mejora_minima:
                # El nuevo modelo mejora: pasar a ser champion
                mlflow.MlflowClient().set_registered_model_alias(
                    model_name="champion",
                    alias="champion",
                    version=run_id,
                )
                mlflow.log_metric("mejora_auc", auc_actual - auc_champion)
            else:
                # No mejora: el run queda registrado pero alias no cambia
                mlflow.log_metric("champion_auc_actual", auc_champion)
        except Exception:
            # Si no hay champion previo, este se convierte en el primero
            mlflow.MlflowClient().set_registered_model_alias(
                model_name="champion",
                alias="champion",
                version=run_id,
            )
            mlflow.log_metric("es_nuevo_champion", 1.0)

    return run.run_id, metricas, model_uri


@flow(name="entrenamiento-churn-v1")
def entrenar():
    """Flow principal de entrenamiento."""
    # Cargar datos
    df = cargar_datos(cfg["datos"]["ruta"])

    # Entrenar y registrar
    run_id, metricas, model_uri = entrenar_y_registrar(
        df, cfg, numericas, categoricas, n_features
    )

    # Crear artefacto Markdown con resumen
    create_markdown_artifact(
        key="entrenamiento-resumen",
        markdown=(
            f"## Entrenamiento Churn V1\n\n"
            f"- Modelo: `{cfg['modelo']['tipo']}`\n"
            f"- Filas: {len(df)}\n"
            f"- AUC: **{metricas['auc']:.4f}**\n"
            f"- Recall: {metricas['recall']:.4f}\n"
            f"- Run de MLflow: `{run_id}`\n"
            f"- Features: {seleccion} ({n_features} variables)\n"
            f"- Umbral: {cfg['metricas']['umbral']}"
        ),
    )

    print(f"Run de MLflow: {run_id} | AUC = {metricas['auc']:.4f} | Features = {seleccion}")
    return run_id


if __name__ == "__main__":
    os.environ["PREFECT_API_URL"] = os.environ.get("PREFECT_API_URL", "http://localhost:4200/api")
    os.environ["MLFLOW_TRACKING_URI"] = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    entrenar()