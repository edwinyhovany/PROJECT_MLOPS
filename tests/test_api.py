"""Tests de la API con un modelo falso (sin MLflow)."""
import sys
from types import SimpleNamespace

import mlflow.sklearn
import mlflow.tracking
import numpy as np
import pytest
from fastapi.testclient import TestClient

from scripts.load_data import COLUMNAS

CLIENTE = {
    "Age": 33,
    "Login_Hour": 1,
    "Session_Duration_Min": 150,
    "Pages_Viewed": 80,
    "Search_Count": 5,
    "Wishlist_Count": 10,
    "Cart_Items": 4,
    "Purchase_Count": 3,
    "Purchase_Value": 2500,
    "Discount_Percentage": 20,
    "Product_Price": 1500,
    "Product_Rating": 4,
    "Product_Popularity": 50,
    "Review_Count": 500,
    "Click_Count": 40,
    "View_Time_Sec": 120,
    "Repeat_Product_View": 2,
    "Browsing_Frequency": 15,
    "Average_Order_Value": 2800,
    "Purchase_Recency_Days": 10,
    "Interaction_Score": 85,
    "Customer_Loyalty_Score": 60,
    "Product_Relevance_Score": 0.85,
    "Gender": "Male",
    "Region": "North",
    "Occupation": "Business",
    "Income_Level": "Middle",
    "Membership_Level": "Silver",
    "Shopping_Device": "Mobile",
    "Coupon_Used": "No",
    "Payment_Method": "Credit Card",
    "Delivery_Preference": "Express",
    "Product_Category": "Electronics",
    "Brand": "Brand_A",
    "Favorite_Category": "Electronics",
}


class ModeloFalso:
    feature_names_in_ = np.array([c for c in COLUMNAS if c != "Recommendation_Label"])

    def predict_proba(self, X):
        return np.tile([0.3, 0.7], (len(X), 1))


@pytest.fixture
def client(monkeypatch):
    # Se parchean los objetos de módulo (no rutas en texto) y después se importa la API:
    # al importarse, main.py carga el modelo desde MLflow.
    monkeypatch.setattr(mlflow.sklearn, "load_model", lambda uri: ModeloFalso())
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", lambda: SimpleNamespace(
        get_model_version_by_alias=lambda nombre, alias: SimpleNamespace(version="7")))
    sys.modules.pop("infraestructure.api.main", None)

    import infraestructure.api.main as main

    return TestClient(main.app)


def test_health_y_predict(client):
    health = client.get("/health").json()
    assert health["estado"] == "ok"
    assert health["modelo"] == "ecommerce-recommendation-modelo"
    assert health["version_champion"] == "7"

    respuesta = client.post("/predict", json=CLIENTE)
    assert respuesta.status_code == 200
    assert respuesta.json() == {
        "probabilidad_recomendacion": 0.7,
        "prediccion": 1,
        "etiqueta": "Recommended",
        "version_modelo": "7",
    }


def test_raiz_describe_el_servicio(client):
    respuesta = client.get("/")
    assert respuesta.status_code == 200
    assert respuesta.json()["version_champion"] == "7"
    assert "/predict" in respuesta.json()["endpoints"]


def test_modelos_expone_el_champion(client):
    respuesta = client.get("/v1/models")
    assert respuesta.status_code == 200
    assert respuesta.json()["data"][0]["version"] == "7"
    assert respuesta.json()["data"][0]["alias"] == "champion"


def test_predecir_es_alias_de_predict(client):
    respuesta = client.post("/predecir", json=CLIENTE)
    assert respuesta.status_code == 200
    assert respuesta.json()["prediccion"] == 1


@pytest.mark.parametrize("campo,valor", [
    ("Age", 5),
    ("Purchase_Value", 1),
    ("Product_Rating", 9),
    ("Product_Relevance_Score", 2),
    ("Region", "Marte"),
    ("Brand", "Brand_Z"),
    ("Product_Category", "Juguetes"),
])
def test_predict_rechaza_datos_invalidos(client, campo, valor):
    payload = {**CLIENTE, campo: valor}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_rechaza_campos_faltantes(client):
    payload = {k: v for k, v in CLIENTE.items() if k != "Age"}
    assert client.post("/predict", json=payload).status_code == 422
