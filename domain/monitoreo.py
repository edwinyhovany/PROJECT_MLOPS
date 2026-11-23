"""Lógica de drift del proyecto: emulación, reporte de features y drift del target.

No importa Prefect ni MLflow. Evidently se importa de forma perezosa dentro de
`calcular_drift_features`, así los tests de la emulación y del target no necesitan cargar
la librería de reportes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split

from domain.modelo import CATEGORICAS, NUMERICAS, OBJETIVO, VARIABLES

# Subida máxima de la tasa de la variable objetivo (en puntos decimales: 0.3 = +30%).
SUBIDA_TASA_MAXIMA = 0.3

# Desplazamiento de las variables numéricas afectadas, en desviaciones típicas.
DESPLAZAMIENTO_NUMERICO = 2.0

# Fracción de filas de una variable categórica que se reasigna cuando esa variable tiene drift.
FRACCION_CATEGORICA = 0.3


def dividir(df: pd.DataFrame, division: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parte los datos igual que el entrenamiento. Devuelve (entrenamiento, prueba)."""
    return train_test_split(
        df,
        test_size=division["test_size"],
        random_state=division["random_state"],
        stratify=df[OBJETIVO],
    )


def aplicar_drift(df: pd.DataFrame, nivel: float, semilla: int = 42) -> pd.DataFrame:
    """Emula drift en un lote de forma controlada y reproducible.

    El nivel (0.0 a 1.0) controla tres cosas:
    1. Fracción de variables numéricas desplazadas: `round(nivel * n_numericas)`. Cada una se
       desplaza `DESPLAZAMIENTO_NUMERICO` desviaciones típicas.
    2. Variables categóricas: cada una tiene probabilidad `nivel * 0.5` de cambiar y, si cambia,
       se reasigna el `nivel * 0.3` de sus filas a otra categoría.
    3. Tasa del target: sube `nivel * SUBIDA_TASA_MAXIMA` puntos convirtiendo negativos en
       positivos al azar. A nivel 0.6 la tasa sube 0.18 puntos (18 puntos porcentuales).
    """
    if not isinstance(nivel, (int, float)) or isinstance(nivel, bool) or not 0 <= nivel <= 1:
        raise ValueError(f"El nivel de drift debe estar entre 0 y 1, no {nivel}")

    out = df.copy()
    rng = np.random.default_rng(semilla)

    # 1. Drift en features numéricas.
    n_numericas = round(nivel * len(NUMERICAS))
    if n_numericas > 0:
        columnas_afectadas = rng.choice(NUMERICAS, size=n_numericas, replace=False)
        for columna in columnas_afectadas:
            if columna not in out.columns:
                continue
            std_val = out[columna].std()
            if pd.notna(std_val) and std_val > 0:
                out[columna] = out[columna] + DESPLAZAMIENTO_NUMERICO * std_val

    # 2. Drift en features categóricas.
    if nivel > 0:
        for columna in CATEGORICAS:
            if columna not in out.columns or out[columna].empty:
                continue
            if rng.random() >= nivel * 0.5:
                continue
            valores = list(pd.Series(out[columna].dropna().unique()))
            if not valores:
                continue
            moda = out[columna].mode(dropna=True)
            candidatos = [v for v in valores if moda.empty or v != moda.iloc[0]] or valores
            nuevo_valor = candidatos[int(rng.integers(0, len(candidatos)))]
            n_cambiar = int(round(nivel * FRACCION_CATEGORICA * len(out)))
            if n_cambiar > 0:
                indices = rng.choice(out.index, size=n_cambiar, replace=False)
                out.loc[indices, columna] = nuevo_valor

    # 3. Drift en el target (etiqueta).
    out[OBJETIVO] = _subir_tasa(out[OBJETIVO], nivel * SUBIDA_TASA_MAXIMA, rng)
    return out


def _subir_tasa(objetivo: pd.Series, subida: float, rng) -> pd.Series:
    """Convierte negativos en positivos (al azar) hasta que la tasa sube `subida` puntos."""
    if subida <= 0:
        return objetivo
    objetivo_int = objetivo.astype(int)
    tasa_actual = float(objetivo_int.mean())
    nueva_tasa = min(tasa_actual + subida, 1.0)

    pos_objetivo = round(nueva_tasa * len(objetivo_int))
    faltan = pos_objetivo - int(objetivo_int.sum())
    if faltan <= 0:
        return objetivo

    subidos = objetivo.copy()
    indices_ceros = objetivo_int.index[objetivo_int == 0]
    faltan = min(faltan, len(indices_ceros))
    if faltan > 0:
        a_cambiar = rng.choice(indices_ceros, size=faltan, replace=False)
        subidos.loc[a_cambiar] = 1
    return subidos


def fraccion_desde_evidently(resultado: dict) -> float:
    """Saca la fracción de variables con drift del resultado de Evidently (`snapshot.dict()`)."""
    for metrica in resultado["metrics"]:
        if metrica["metric_name"].startswith("DriftedColumnsCount"):
            valor = metrica["value"]
            return float(valor["share"] if isinstance(valor, dict) else valor)
    raise ValueError("El resultado de Evidently no incluye DriftedColumnsCount")


def calcular_drift_features(
    referencia: pd.DataFrame,
    lote: pd.DataFrame,
    ruta_reporte: str | Path | None = None,
    columnas: list[str] | None = None,
):
    """Calcula el drift de features con Evidently y, si se indica, guarda el reporte HTML.

    Devuelve `(fraccion, resultado)`, donde `fraccion` es la proporción de variables con
    drift y `resultado` es el snapshot de Evidently.
    """
    from evidently import Report
    from evidently.presets import DataDriftPreset

    columnas = columnas or VARIABLES
    faltan = [c for c in columnas if c not in referencia.columns or c not in lote.columns]
    if faltan:
        raise ValueError(f"Faltan variables para el drift de features: {faltan}")

    resultado = Report([DataDriftPreset()]).run(
        reference_data=referencia[columnas],
        current_data=lote[columnas],
    )
    if ruta_reporte is not None:
        ruta_reporte = Path(ruta_reporte)
        ruta_reporte.parent.mkdir(parents=True, exist_ok=True)
        resultado.save_html(str(ruta_reporte))
    return fraccion_desde_evidently(resultado.dict()), resultado


def hay_alerta_features(fraccion: float, umbral: float) -> bool:
    """Hay alerta si la fracción de variables con drift supera el umbral configurado."""
    if not 0 <= umbral <= 1:
        raise ValueError(f"El umbral de drift debe estar entre 0 y 1, no {umbral}")
    return bool(fraccion > umbral)


def validar_lote(lote: pd.DataFrame) -> None:
    """Falla si el lote no trae la variable objetivo con valores binarios."""
    if OBJETIVO not in lote.columns:
        raise ValueError(
            f"El lote no tiene la columna '{OBJETIVO}': no se puede calcular el drift de evento. "
            "Las etiquetas reales pueden llegar con retraso; vuelve a ejecutar el flow cuando estén."
        )
    if lote[OBJETIVO].isna().any():
        raise ValueError(
            f"El lote tiene '{OBJETIVO}' sin valor: no se puede calcular el drift de evento. "
            "Las etiquetas reales pueden llegar con retraso."
        )
    valores = set(pd.to_numeric(lote[OBJETIVO], errors="coerce").dropna().unique())
    if not valores <= {0, 1}:
        raise ValueError(f"El lote tiene valores no binarios en '{OBJETIVO}': {sorted(valores)}")


def drift_de_target(referencia: pd.Series, lote: pd.Series, alfa: float) -> dict:
    """Compara la proporción de la variable objetivo entre referencia y lote (chi-cuadrado)."""
    if len(referencia) == 0 or len(lote) == 0:
        raise ValueError("La referencia y el lote deben tener al menos una fila")
    if not 0 < alfa < 1:
        raise ValueError(f"El nivel de significación debe estar entre 0 y 1, no {alfa}")

    ref_int = referencia.astype(int)
    lote_int = lote.astype(int)

    pos_ref = int(ref_int.sum())
    neg_ref = len(ref_int) - pos_ref
    pos_lote = int(lote_int.sum())
    neg_lote = len(lote_int) - pos_lote

    tabla = [[pos_ref, neg_ref], [pos_lote, neg_lote]]
    if sum(tabla[0]) == 0 or sum(tabla[1]) == 0:
        raise ValueError("La referencia y el lote no pueden estar vacíos")
    if pos_ref + pos_lote == 0:
        p_valor = 1.0
    elif neg_ref + neg_lote == 0:
        p_valor = 1.0
    else:
        try:
            _, p_valor, _, _ = chi2_contingency(tabla)
            p_valor = float(p_valor)
        except ValueError:
            # Tabla degenerada: una clase solo aparece en una de las dos muestras.
            p_valor = 0.0

    return {
        "tasa_referencia": pos_ref / len(ref_int),
        "tasa_lote": pos_lote / len(lote_int),
        "p_valor": float(p_valor),
        "drift": bool(p_valor < alfa),
    }


def hay_alerta_evento(resultado: dict, alfa: float) -> bool:
    """Hay alerta de evento si el p-valor del contraste queda por debajo de alfa."""
    return bool(resultado["p_valor"] < alfa)


def guardar_resultado_evento(resultado: dict, ruta: str | Path, extra: dict | None = None) -> Path:
    """Guarda el resultado del drift de evento en JSON para dejar evidencia por lote."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    contenido = {"target": OBJETIVO, **resultado, **(extra or {})}
    ruta.write_text(json.dumps(contenido, indent=2, ensure_ascii=False), encoding="utf-8")
    return ruta
