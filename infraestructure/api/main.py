"""API FastAPI que sirve el modelo champion (Actividad 2).

- Carga el modelo champion desde el Model Registry de MLflow.
- Expone endpoints:
  - GET /: información general de la API.
  - GET /health: estado del servicio y versión activa del champion.
  - GET /v1/models: metadatos del modelo champion activo (compatibilidad estándar).
  - POST /predict: recibe los datos de un cliente y devuelve probabilidad y predicción.
  - POST /predecir: alias en español para /predict.
"""

import os
import sys
from pathlib import Path

# Raíz del proyecto en sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mlflow
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

import domain.modelo as modelo_domain

CONFIG_FILE = ROOT_DIR / "configs" / "config.yaml"

with open(CONFIG_FILE, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

NOMBRE_MODELO = cfg["registro"]["nombre_modelo"]
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"])
mlflow.set_tracking_uri(TRACKING_URI)

app = FastAPI(
    title="API de Recomendación E-Commerce",
    description="Servicio de inferencia FastAPI que sirve el modelo champion de MLflow",
    version="1.0.0",
)

# Cargar modelo champion al arrancar
modelo = None
version_champion = None

try:
    client = MlflowClient()
    v_info = client.get_model_version_by_alias(NOMBRE_MODELO, "champion")
    version_champion = v_info.version
    modelo = mlflow.sklearn.load_model(f"models:/{NOMBRE_MODELO}@champion")
    print(f"Modelo champion cargado: {NOMBRE_MODELO} (versión {version_champion})")
except Exception as e:
    print(f"Aviso al iniciar API: no se pudo cargar champion ({e}). Se reintentará bajo demanda.")


class ClienteRequest(BaseModel):
    """Datos de cliente según el dataset Consumer Portrait E-Commerce."""
    Age: float = 33.0
    Login_Hour: float = 1.0
    Session_Duration_Min: float = 150.0
    Pages_Viewed: float = 80.0
    Search_Count: float = 5.0
    Wishlist_Count: float = 10.0
    Cart_Items: float = 4.0
    Purchase_Count: float = 3.0
    Purchase_Value: float = 2500.0
    Discount_Percentage: float = 20.0
    Product_Price: float = 1500.0
    Product_Rating: float = 4.0
    Product_Popularity: float = 50.0
    Review_Count: float = 500.0
    Click_Count: float = 40.0
    View_Time_Sec: float = 120.0
    Repeat_Product_View: float = 2.0
    Browsing_Frequency: float = 15.0
    Average_Order_Value: float = 2800.0
    Purchase_Recency_Days: float = 10.0
    Interaction_Score: float = 85.0
    Customer_Loyalty_Score: float = 60.0
    Product_Relevance_Score: float = 0.85
    Gender: str = "Male"
    Region: str = "North"
    Occupation: str = "Business"
    Income_Level: str = "Middle"
    Membership_Level: str = "Silver"
    Shopping_Device: str = "Mobile"
    Coupon_Used: str = "No"
    Payment_Method: str = "Credit Card"
    Delivery_Preference: str = "Express"
    Product_Category: str = "Electronics"
    Brand: str = "Brand_A"
    Favorite_Category: str = "Electronics"


class PrediccionResponse(BaseModel):
    probabilidad_recomendacion: float
    prediccion: int
    etiqueta: str
    version_modelo: str


@app.get("/")
def root():
    return {
        "mensaje": "API de Recomendación E-Commerce",
        "modelo": NOMBRE_MODELO,
        "version_champion": version_champion,
        "estado": "listo" if modelo is not None else "sin_modelo_champion",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "estado": "ok" if modelo is not None else "sin_modelo_champion",
        "modelo": NOMBRE_MODELO,
        "version_champion": version_champion,
        "tracking_uri": TRACKING_URI,
    }


@app.get("/v1/models")
def get_models():
    """Endpoint de compatibilidad estándar para inspección de modelos."""
    return {
        "data": [
            {
                "id": NOMBRE_MODELO,
                "alias": "champion",
                "version": version_champion,
                "status": "ready" if modelo is not None else "loading",
            }
        ]
    }


def _ejecutar_prediccion(cliente: ClienteRequest):
    global modelo, version_champion
    if modelo is None:
        try:
            client = MlflowClient()
            v_info = client.get_model_version_by_alias(NOMBRE_MODELO, "champion")
            version_champion = v_info.version
            modelo = mlflow.sklearn.load_model(f"models:/{NOMBRE_MODELO}@champion")
        except Exception as err:
            raise HTTPException(
                status_code=503,
                detail=f"Modelo champion no disponible en MLflow: {err}. Ejecute primero el pipeline de entrenamiento.",
            )

    datos_dict = cliente.model_dump()
    df_in = pd.DataFrame([datos_dict])

    cols_modelo = list(getattr(modelo, "feature_names_in_", []))
    if not cols_modelo:
        cols_modelo = modelo_domain.VARIABLES

    X = df_in[[c for c in cols_modelo if c in df_in.columns]]
    proba = float(modelo.predict_proba(X)[0, 1])
    umbral = cfg.get("metricas", {}).get("umbral", 0.5)
    pred = int(proba >= umbral)

    return {
        "probabilidad_recomendacion": round(proba, 4),
        "prediccion": pred,
        "etiqueta": "Recommended" if pred == 1 else "Not Recommended / Neutral",
        "version_modelo": str(version_champion),
    }


@app.post("/predict", response_model=PrediccionResponse)
def predict(cliente: ClienteRequest):
    return _ejecutar_prediccion(cliente)


@app.post("/predecir", response_model=PrediccionResponse)
def predecir(cliente: ClienteRequest):
    return _ejecutar_prediccion(cliente)
