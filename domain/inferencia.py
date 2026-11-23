"""Lógica pura de la pipeline de inferencia con drift emulado.

La inferencia toma un lote de la parte de prueba, le aplica el drift configurado, lo puntúa
y guarda una copia para que los pipelines de drift la lean. Nada de esto importa Prefect ni
MLflow, así que se puede probar con un modelo falso o entrenado en memoria.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from domain.modelo import OBJETIVO
from domain.monitoreo import aplicar_drift, dividir


def tomar_lote(df: pd.DataFrame, division: dict, tamano_lote: int, semilla: int = 42) -> pd.DataFrame:
    """Toma una muestra de la parte de prueba del dataset (sin aplicar drift)."""
    if tamano_lote <= 0:
        raise ValueError("El tamaño del lote debe ser mayor que cero")
    _, prueba = dividir(df, division)
    n = min(tamano_lote, len(prueba))
    return prueba.sample(n=n, random_state=semilla).copy()


def preparar_lote(
    df: pd.DataFrame,
    division: dict,
    tamano_lote: int,
    nivel_drift: float,
    semilla: int = 42,
) -> pd.DataFrame:
    """Toma un lote de prueba y le aplica el drift emulado del nivel indicado."""
    base = tomar_lote(df, division, tamano_lote, semilla)
    return aplicar_drift(base, nivel_drift, semilla)


def puntuar(modelo, lote: pd.DataFrame, umbral: float = 0.5) -> pd.DataFrame:
    """Añade `probabilidad_recomendacion` y `prediccion` al lote usando las columnas del modelo."""
    columnas = list(getattr(modelo, "feature_names_in_", [c for c in lote.columns if c != OBJETIVO]))
    proba = modelo.predict_proba(lote[columnas])[:, 1]
    resultado = lote.copy()
    resultado["probabilidad_recomendacion"] = proba
    resultado["prediccion"] = (proba >= umbral).astype(int)
    return resultado


def nombre_lote(nivel: float, fecha: datetime | None = None) -> str:
    """Nombre del archivo histórico del lote: `lote_<fecha>_nivel_<nivel>.csv`."""
    fecha = fecha or datetime.now()
    return f"lote_{fecha:%Y%m%d_%H%M%S}_nivel_{nivel:.1f}.csv"


def guardar_lote(
    lote: pd.DataFrame,
    carpeta: str | Path,
    nivel: float,
    fecha: datetime | None = None,
) -> dict[str, str]:
    """Guarda el lote en tres rutas: histórica, última por nivel y lote actual.

    - `lote_<fecha>_nivel_<nivel>.csv`: copia de cada ejecución (no se sobrescribe).
    - `lote_nivel_<nivel>.csv`: última ejecución de ese nivel, cómoda para comparar.
    - `lote_actual.csv`: la última ejecución, la que leen los flows de drift programados.
    """
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    fecha = fecha or datetime.now()

    ruta_historica = carpeta / nombre_lote(nivel, fecha)
    ruta_nivel = carpeta / f"lote_nivel_{nivel:.1f}.csv"
    ruta_actual = carpeta / "lote_actual.csv"

    for ruta in (ruta_historica, ruta_nivel, ruta_actual):
        lote.to_csv(ruta, index=False)

    return {
        "historico": str(ruta_historica),
        "nivel": str(ruta_nivel),
        "actual": str(ruta_actual),
    }


def leer_lote(ruta: str | Path) -> pd.DataFrame:
    """Lee un lote guardado en CSV."""
    return pd.read_csv(ruta)
