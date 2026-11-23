"""Tests de drift: emulación, reporte de features y drift del target. Sin servicios."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from domain.modelo import CATEGORICAS, NUMERICAS, OBJETIVO
from domain.monitoreo import (
    SUBIDA_TASA_MAXIMA,
    aplicar_drift,
    calcular_drift_features,
    drift_de_target,
    fraccion_desde_evidently,
    guardar_resultado_evento,
    hay_alerta_evento,
    hay_alerta_features,
    validar_lote,
)
from scripts.load_data import cargar_datos


@pytest.fixture
def datos():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({columna: rng.normal(0, 1, 500) for columna in NUMERICAS})
    df[OBJETIVO] = rng.integers(0, 2, 500)
    return df


@pytest.fixture(scope="module")
def datos_reales():
    return cargar_datos().sample(2000, random_state=21)


def test_drift_cero_no_cambia_los_datos(datos):
    assert aplicar_drift(datos, 0.0).equals(datos)


def test_drift_sube_la_tasa_del_objetivo_en_proporcion_al_nivel(datos):
    tasa = datos[OBJETIVO].mean()
    lote = aplicar_drift(datos, 0.6)
    assert lote[OBJETIVO].mean() == pytest.approx(tasa + 0.6 * SUBIDA_TASA_MAXIMA, abs=0.01)


def test_drift_cambia_la_fraccion_indicada_de_variables(datos):
    lote = aplicar_drift(datos, 0.6)
    cambiadas = [c for c in NUMERICAS if not np.allclose(lote[c], datos[c])]
    assert len(cambiadas) == round(0.6 * len(NUMERICAS))


def test_drift_alto_cambia_alguna_variable_categorica(datos_reales):
    lote = aplicar_drift(datos_reales, 0.6, semilla=21)
    cambiadas = [c for c in CATEGORICAS if not lote[c].equals(datos_reales[c])]
    assert cambiadas, "El drift 0.6 debe afectar también a alguna variable categórica"


def test_drift_rechaza_nivel_fuera_de_rango(datos):
    for nivel in (-0.1, 1.1, "alto", None):
        with pytest.raises(ValueError):
            aplicar_drift(datos, nivel)


def test_fraccion_desde_evidently():
    resultado = {"metrics": [
        {"metric_name": "ValueDrift(column=Age)", "value": 0.5},
        {"metric_name": "DriftedColumnsCount(drift_share=0.5)", "value": {"count": 7, "share": 0.5}},
    ]}
    assert fraccion_desde_evidently(resultado) == 0.5


def test_fraccion_desde_evidently_sin_metrica_falla():
    with pytest.raises(ValueError):
        fraccion_desde_evidently({"metrics": []})


def test_validar_lote_exige_objetivo(datos):
    with pytest.raises(ValueError, match=OBJETIVO):
        validar_lote(datos.drop(columns=[OBJETIVO]))


def test_validar_lote_rechaza_etiquetas_nulas(datos):
    roto = datos.copy()
    roto.loc[roto.index[0], OBJETIVO] = np.nan
    with pytest.raises(ValueError, match="sin valor"):
        validar_lote(roto)


def test_validar_lote_rechaza_etiquetas_no_binarias(datos):
    roto = datos.copy()
    roto.loc[roto.index[0], OBJETIVO] = 2
    with pytest.raises(ValueError, match="no binarios"):
        validar_lote(roto)


def test_drift_de_target_detecta_cambio_de_tasa():
    referencia = pd.Series([0] * 800 + [1] * 200)
    lote = pd.Series([0] * 400 + [1] * 600)
    resultado = drift_de_target(referencia, lote, alfa=0.05)
    assert resultado["drift"] is True
    assert resultado["tasa_referencia"] == pytest.approx(0.2)
    assert resultado["tasa_lote"] == pytest.approx(0.6)
    assert 0 <= resultado["p_valor"] <= 1


def test_drift_de_target_no_alerta_con_la_misma_tasa():
    referencia = pd.Series([0] * 800 + [1] * 200)
    lote = pd.Series([0] * 80 + [1] * 20)
    assert drift_de_target(referencia, lote, alfa=0.05)["drift"] is False


def test_drift_de_target_rechaza_alfa_invalido():
    with pytest.raises(ValueError):
        drift_de_target(pd.Series([0, 1]), pd.Series([0, 1]), alfa=0)


def test_hay_alerta_features():
    assert hay_alerta_features(0.6, 0.3) is True
    assert hay_alerta_features(0.1, 0.3) is False
    assert hay_alerta_features(0.3, 0.3) is False
    with pytest.raises(ValueError):
        hay_alerta_features(0.5, 1.5)


def test_hay_alerta_evento():
    assert hay_alerta_evento({"p_valor": 0.01}, 0.05) is True
    assert hay_alerta_evento({"p_valor": 0.2}, 0.05) is False


def test_guardar_resultado_evento(tmp_path):
    ruta = guardar_resultado_evento(
        {"tasa_referencia": 0.2, "tasa_lote": 0.4, "p_valor": 0.001, "drift": True},
        tmp_path / "evento.json",
        extra={"lote": "lote.csv"},
    )
    assert ruta.exists()
    contenido = ruta.read_text(encoding="utf-8")
    assert '"drift": true' in contenido
    assert '"lote": "lote.csv"' in contenido


def test_reportes_de_features_0_0_y_0_6_son_distintos(datos_reales, tmp_path):
    """El lote 0.0 no debe superar el umbral; el 0.6 sí, y sus reportes difieren."""
    from domain.inferencia import preparar_lote

    division = {"test_size": 0.2, "random_state": 42}
    referencia = datos_reales.sample(frac=0.8, random_state=1)
    lote_0 = preparar_lote(datos_reales, division, 500, nivel_drift=0.0, semilla=7)
    lote_6 = preparar_lote(datos_reales, division, 500, nivel_drift=0.6, semilla=7)

    ruta_0 = tmp_path / "drift_0_0.html"
    ruta_6 = tmp_path / "drift_0_6.html"
    fraccion_0, _ = calcular_drift_features(referencia, lote_0, ruta_0)
    fraccion_6, _ = calcular_drift_features(referencia, lote_6, ruta_6)

    assert ruta_0.exists() and ruta_6.exists()
    assert fraccion_6 > 0.3
    assert fraccion_0 <= 0.3
    assert fraccion_6 > fraccion_0
    assert ruta_0.read_text(encoding="utf-8") != ruta_6.read_text(encoding="utf-8")


def test_calcular_drift_features_rechaza_columnas_faltantes():
    with pytest.raises(ValueError, match="Faltan variables"):
        calcular_drift_features(pd.DataFrame({"Age": [1, 2]}), pd.DataFrame({"Age": [1, 2]}))
