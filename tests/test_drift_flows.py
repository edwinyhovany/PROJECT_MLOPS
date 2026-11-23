"""Tests de la orquestación de drift sin MLflow ni servidores de Prefect.

Se llaman las funciones internas de las tareas (`Task.fn`) y la lógica pura de `domain`.
"""
from pathlib import Path

import pandas as pd
import pytest

from domain.modelo import OBJETIVO
from domain.monitoreo import validar_lote
from infraestructure.flows.drift_evento_flow import evaluar_evento
from scripts.load_data import cargar_datos


@pytest.fixture
def cfg(tmp_path):
    return {"monitoreo": {"carpeta_reportes": str(tmp_path / "reports"), "alfa_evento": 0.05}}


def test_evaluar_evento_guarda_json_y_detecta_drift(cfg, tmp_path):
    datos = cargar_datos().sample(1000, random_state=5)
    referencia = datos[OBJETIVO].head(800)
    lote = datos.tail(200).copy()
    lote[OBJETIVO] = 1  # tasa muy distinta a la referencia
    ruta = tmp_path / "lote_con_drift.csv"
    lote.to_csv(ruta, index=False)

    resultado = evaluar_evento.fn(referencia, str(ruta), cfg)

    assert resultado["drift"] is True
    assert resultado["tasa_lote"] == 1.0
    reporte = Path(cfg["monitoreo"]["carpeta_reportes"]) / "drift_evento_lote_con_drift.json"
    assert reporte.exists()
    assert '"drift": true' in reporte.read_text(encoding="utf-8")


def test_evaluar_evento_rechaza_lote_sin_etiquetas(cfg, tmp_path):
    datos = cargar_datos().head(100).drop(columns=[OBJETIVO])
    ruta = tmp_path / "lote_sin_etiquetas.csv"
    datos.to_csv(ruta, index=False)

    with pytest.raises(ValueError, match="no tiene la columna"):
        evaluar_evento.fn(pd.Series([0, 1]), str(ruta), cfg)


def test_evaluar_evento_rechaza_etiquetas_nulas(cfg, tmp_path):
    datos = cargar_datos().head(100).copy()
    datos.loc[datos.index[0], OBJETIVO] = None
    ruta = tmp_path / "lote_con_nulos.csv"
    datos.to_csv(ruta, index=False)

    with pytest.raises(ValueError, match="sin valor"):
        evaluar_evento.fn(pd.Series([0, 1]), str(ruta), cfg)


def test_validar_lote_mensaje_claro_sin_etiquetas():
    with pytest.raises(ValueError, match="etiquetas reales pueden llegar con retraso"):
        validar_lote(pd.DataFrame({"Age": [20, 30]}))
