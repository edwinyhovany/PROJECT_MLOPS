"""Tests del modelo: features, pipeline, métricas y algoritmos. Sin MLflow ni Prefect."""
import numpy as np
import pandas as pd
import pytest

from domain.modelo import (
    CATEGORICAS,
    NUMERICAS,
    binarizar_objetivo,
    calcular_metricas,
    construir_modelo,
    resolver_features,
)


def test_resolver_features_all():
    numericas, categoricas = resolver_features("ALL")
    assert numericas == NUMERICAS
    assert categoricas == CATEGORICAS


def test_resolver_features_lista_separa_numericas_y_categoricas():
    numericas, categoricas = resolver_features(["Age", "Gender", "Region"])
    assert numericas == ["Age"]
    assert categoricas == ["Gender", "Region"]


def test_resolver_features_rechaza_variables_desconocidas():
    with pytest.raises(ValueError):
        resolver_features(["NoExiste"])


def test_resolver_features_rechaza_seleccion_vacia():
    with pytest.raises(ValueError):
        resolver_features([])


@pytest.mark.parametrize("tipo, parametros", [
    ("logistic_regression", {"C": 1.0, "max_iter": 1000}),
    ("gradient_boosting", {"n_estimators": 10, "max_depth": 2, "random_state": 42}),
    ("random_forest", {"n_estimators": 10, "max_depth": 3, "random_state": 42}),
])
def test_modelo_predice_probabilidades_validas(tipo, parametros):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "Age": rng.integers(18, 70, 200),
        "Purchase_Value": rng.random(200) * 5000,
        "Gender": rng.choice(["Male", "Female"], 200),
        "Region": rng.choice(["North", "South", "East"], 200),
    })
    y = (X["Age"] > 40).astype(int)

    modelo = construir_modelo(tipo, parametros, ["Age", "Purchase_Value"], ["Gender", "Region"])
    modelo.fit(X, y)
    proba = modelo.predict_proba(X)[:, 1]

    assert proba.shape == (200,)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_modelo_desconocido_falla():
    with pytest.raises(ValueError, match="no soportado"):
        construir_modelo("red_neuronal", {}, ["Age"], [])


def test_calcular_metricas_perfecto():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])
    metricas = calcular_metricas(y, proba, umbral=0.5)
    assert metricas["auc"] == 1.0
    assert metricas["accuracy"] == 1.0
    assert metricas["precision"] == 1.0
    assert metricas["recall"] == 1.0
    assert metricas["f1"] == 1.0


def test_calcular_metricas_respeta_el_umbral():
    y = np.array([0, 1])
    proba = np.array([0.4, 0.6])
    assert calcular_metricas(y, proba, umbral=0.5)["accuracy"] == 1.0
    assert calcular_metricas(y, proba, umbral=0.7)["accuracy"] == 0.5


def test_binarizar_objetivo_mapea_recommended():
    serie = pd.Series(["Recommended", "Neutral", "Highly Recommended", "Low Priority"])
    assert binarizar_objetivo(serie).tolist() == [1, 0, 0, 0]


def test_binarizar_objetivo_es_idempotente():
    serie = pd.Series([0, 1, 1, 0])
    pd.testing.assert_series_equal(binarizar_objetivo(serie), serie)
