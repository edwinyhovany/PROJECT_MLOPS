"""Tests de la lógica del pipeline de entrenamiento, sin MLflow ni Prefect."""
import numpy as np
import pytest

from domain.entrenamiento import (
    ResultadoEntrenamiento,
    decidir_promocion,
    dividir_datos,
    elegir_mejor,
    entrenar_y_evaluar,
    preparar_matrices,
    resumen_resultados,
)
from domain.modelo import OBJETIVO
from scripts.load_data import cargar_datos

FEATURES = {
    "nombre": "custom",
    "columnas": [
        "Age", "Gender", "Region", "Purchase_Count", "Purchase_Value",
        "Interaction_Score", "Customer_Loyalty_Score", "Product_Relevance_Score",
    ],
}
DIVISION = {"test_size": 0.2, "random_state": 42}


@pytest.fixture(scope="module")
def datos():
    return cargar_datos().sample(1500, random_state=3)


def test_dividir_datos_respeta_tamano_y_estratificacion(datos):
    train, test = dividir_datos(datos, DIVISION)
    assert len(train) + len(test) == len(datos)
    assert len(test) == round(len(datos) * DIVISION["test_size"])
    assert abs(train[OBJETIVO].mean() - test[OBJETIVO].mean()) < 0.05


def test_preparar_matrices_usa_las_variables_elegidas(datos):
    train, test = dividir_datos(datos, DIVISION)
    X_train, y_train, X_test, y_test, variables = preparar_matrices(train, test, FEATURES)
    assert set(variables) == set(FEATURES["columnas"])
    assert list(X_train.columns) == variables
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_entrenar_y_evaluar_devuelve_metricas_y_modelo(datos):
    resultado = entrenar_y_evaluar(
        datos,
        DIVISION,
        FEATURES,
        "logistic_regression",
        {"C": 1.0, "max_iter": 1000},
        umbral=0.5,
        nombre_run="prueba",
    )
    assert resultado.nombre_run == "prueba"
    assert set(resultado.metricas) == {"auc", "accuracy", "precision", "recall", "f1"}
    assert 0 <= resultado.auc <= 1
    assert hasattr(resultado.modelo, "predict_proba")


def test_elegir_mejor_por_auc():
    resultados = [
        ResultadoEntrenamiento("a", "all", "logistic_regression", [], None, {"auc": 0.7}),
        ResultadoEntrenamiento("b", "all", "random_forest", [], None, {"auc": 0.9}),
        ResultadoEntrenamiento("c", "all", "gradient_boosting", [], None, {"auc": 0.8}),
    ]
    assert elegir_mejor(resultados).nombre_run == "b"


def test_elegir_mejor_sin_resultados_falla():
    with pytest.raises(ValueError):
        elegir_mejor([])


@pytest.mark.parametrize("auc,umbral,champion,esperado", [
    (0.95, 0.90, None, True),
    (0.85, 0.90, None, False),
    (0.99, 0.90, 0.98, True),
    (0.98, 0.90, 0.98, False),
    (0.97, 0.90, 0.98, False),
])
def test_decidir_promocion(auc, umbral, champion, esperado):
    promover, motivo = decidir_promocion(auc, umbral, champion)
    assert promover is esperado
    assert isinstance(motivo, str) and motivo


def test_resumen_resultados_es_serializable():
    resultados = [
        ResultadoEntrenamiento(
            "run-1", "all", "logistic_regression", ["Age", "Gender"], None,
            {"auc": 0.8, "accuracy": 0.7, "precision": 0.6, "recall": 0.5, "f1": 0.55},
        )
    ]
    resumen = resumen_resultados(resultados)
    assert resumen[0]["nombre_run"] == "run-1"
    assert resumen[0]["n_features"] == 2
    assert np.isclose(resumen[0]["auc"], 0.8)
