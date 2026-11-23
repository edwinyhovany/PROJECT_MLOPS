"""Tests de la lógica de inferencia con drift, sin MLflow ni Prefect."""
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from domain.inferencia import guardar_lote, leer_lote, nombre_lote, preparar_lote, puntuar, tomar_lote
from domain.modelo import NUMERICAS, OBJETIVO
from scripts.load_data import COLUMNAS, cargar_datos

DIVISION = {"test_size": 0.2, "random_state": 42}
FEATURES = [c for c in COLUMNAS if c != OBJETIVO]


class ModeloFalso:
    feature_names_in_ = np.array(FEATURES)

    def predict_proba(self, X):
        return np.tile([0.4, 0.6], (len(X), 1))


@pytest.fixture(scope="module")
def datos():
    return cargar_datos().sample(2000, random_state=5)


def test_tomar_lote_tamano_y_columnas(datos):
    lote = tomar_lote(datos, DIVISION, 100, semilla=5)
    assert len(lote) == 100
    assert list(lote.columns) == list(datos.columns)


def test_tomar_lote_es_reproducible(datos):
    a = tomar_lote(datos, DIVISION, 100, semilla=5)
    b = tomar_lote(datos, DIVISION, 100, semilla=5)
    assert a.equals(b)


def test_tomar_lote_rechaza_tamano_invalido(datos):
    with pytest.raises(ValueError):
        tomar_lote(datos, DIVISION, 0)


def test_preparar_lote_nivel_cero_no_cambia_el_lote(datos):
    base = tomar_lote(datos, DIVISION, 100, semilla=5)
    lote = preparar_lote(datos, DIVISION, 100, nivel_drift=0.0, semilla=5)
    assert base.equals(lote)


def test_preparar_lote_nivel_alto_cambia_features_y_target(datos):
    base = tomar_lote(datos, DIVISION, 300, semilla=5)
    lote = preparar_lote(datos, DIVISION, 300, nivel_drift=0.6, semilla=5)
    cambiadas = [c for c in NUMERICAS if not np.allclose(base[c], lote[c])]
    assert len(cambiadas) == round(0.6 * len(NUMERICAS))
    assert lote[OBJETIVO].mean() > base[OBJETIVO].mean()


def test_preparar_lote_rechaza_nivel_invalido(datos):
    with pytest.raises(ValueError):
        preparar_lote(datos, DIVISION, 100, nivel_drift=1.5)


def test_puntuar_anade_probabilidad_y_prediccion(datos):
    lote = tomar_lote(datos, DIVISION, 50, semilla=5)
    resultado = puntuar(ModeloFalso(), lote, umbral=0.5)
    assert list(resultado.columns[-2:]) == ["probabilidad_recomendacion", "prediccion"]
    assert (resultado["probabilidad_recomendacion"] == 0.6).all()
    assert (resultado["prediccion"] == 1).all()


def test_puntuar_respeta_el_umbral(datos):
    lote = tomar_lote(datos, DIVISION, 10, semilla=5)
    assert (puntuar(ModeloFalso(), lote, umbral=0.7)["prediccion"] == 0).all()


def test_guardar_lote_escribe_historico_nivel_y_actual(tmp_path, datos):
    lote = preparar_lote(datos, DIVISION, 100, nivel_drift=0.3, semilla=5)
    rutas = guardar_lote(lote, tmp_path, nivel=0.3)
    for ruta in rutas.values():
        assert Path(ruta).exists()
    actual = leer_lote(rutas["actual"])
    assert len(actual) == 100
    assert "nivel_0.3" in rutas["nivel"]
    assert "nivel_0.3" in rutas["historico"]


def test_nombre_lote_incluye_nivel_y_fecha():
    assert nombre_lote(0.6, datetime(2026, 1, 2, 3, 4, 5)) == "lote_20260102_030405_nivel_0.6.csv"
