"""API FastAPI que sirve el modelo champion de recomendación E-Commerce (Actividad 3).

- Carga el modelo champion desde el Model Registry de MLflow.
- Expone endpoints:
  - GET /: información general de la API.
  - GET /health: estado del servicio y versión activa del champion.
  - GET /v1/models: metadatos del modelo champion activo.
  - POST /predict: recibe los datos de un cliente y devuelve probabilidad y predicción.
  - POST /predecir: alias en español para /predict.

El payload se valida con Pydantic: rangos numéricos y categorías del dataset. Un payload
inválido devuelve 422. La carga del champion se hace al arrancar; si MLflow no está
disponible, la API arranca igualmente y `/predict` devuelve 503 hasta que el champion exista.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

# Raíz del proyecto en sys.path.
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

import domain.modelo as modelo_domain
from scripts.load_data import cargar_config

cfg = cargar_config()
NOMBRE_MODELO = cfg["registro"]["nombre_modelo"]
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"])
mlflow.set_tracking_uri(TRACKING_URI)

app = FastAPI(
    title="API de Recomendación E-Commerce",
    description="Servicio de inferencia FastAPI que sirve el modelo champion de MLflow (Actividad 3)",
    version="3.0.0",
)

# Cargar modelo champion al arrancar.
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
    """Datos de un cliente según el dataset Consumer Portrait E-Commerce (35 variables)."""

    # Numéricas
    Age: float = Field(ge=18, le=70)
    Login_Hour: float = Field(ge=0, le=23)
    Session_Duration_Min: float = Field(ge=2, le=180)
    Pages_Viewed: float = Field(ge=1, le=100)
    Search_Count: float = Field(ge=0, le=60)
    Wishlist_Count: float = Field(ge=0, le=30)
    Cart_Items: float = Field(ge=0, le=20)
    Purchase_Count: float = Field(ge=0, le=10)
    Purchase_Value: float = Field(ge=20, le=5000)
    Discount_Percentage: float = Field(ge=0, le=80)
    Product_Price: float = Field(ge=10, le=5000)
    Product_Rating: float = Field(ge=1, le=5)
    Product_Popularity: float = Field(ge=1, le=100)
    Review_Count: float = Field(ge=0, le=5000)
    Click_Count: float = Field(ge=0, le=150)
    View_Time_Sec: float = Field(ge=5, le=900)
    Repeat_Product_View: float = Field(ge=0, le=20)
    Browsing_Frequency: float = Field(ge=1, le=100)
    Average_Order_Value: float = Field(ge=20, le=5000)
    Purchase_Recency_Days: float = Field(ge=0, le=365)
    Interaction_Score: float = Field(ge=0, le=100)
    Customer_Loyalty_Score: float = Field(ge=0, le=100)
    Product_Relevance_Score: float = Field(ge=0, le=1)

    # Categóricas
    Gender: Literal["Male", "Female"]
    Region: Literal["North", "Central", "West", "Metro", "South", "Coastal", "Northeast", "East"]
    Occupation: Literal[
        "Business", "Student", "Retired", "Freelancer", "Government",
        "Doctor", "Homemaker", "Engineer", "Self-Employed", "Teacher",
    ]
    Income_Level: Literal["Low", "Upper-Middle", "Middle", "Lower-Middle", "High"]
    Membership_Level: Literal["Basic", "Silver", "Platinum", "Gold"]
    Shopping_Device: Literal["Mobile", "Desktop", "Tablet"]
    Coupon_Used: Literal["No", "Yes"]
    Payment_Method: Literal[
        "Net Banking", "Debit Card", "Cash on Delivery",
        "Digital Wallet", "Credit Card", "UPI",
    ]
    Delivery_Preference: Literal["Express", "Standard", "Same Day"]
    Product_Category: Literal[
        "Electronics", "Health", "Home Decor", "Fashion", "Toys", "Furniture",
        "Sports", "Beauty", "Jewelry", "Books", "Automotive", "Grocery",
    ]
    Brand: Literal[
        "Brand_A", "Brand_B", "Brand_C", "Brand_D", "Brand_E",
        "Brand_F", "Brand_G", "Brand_H", "Brand_I", "Brand_J",
    ]
    Favorite_Category: Literal[
        "Electronics", "Health", "Home Decor", "Fashion", "Toys", "Furniture",
        "Sports", "Beauty", "Jewelry", "Books", "Automotive", "Grocery",
    ]


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
        "endpoints": ["/health", "/v1/models", "/predict", "/docs"],
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
