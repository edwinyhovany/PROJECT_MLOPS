"""Lógica pura del pipeline de entrenamiento.

Contiene la partición, el entrenamiento, la evaluación y la decisión de promoción.
No importa Prefect ni MLflow: el flow solo orquesta y registra. Así se puede probar todo
sin servicios levantados.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from domain.modelo import OBJETIVO, calcular_metricas, construir_modelo, resolver_features


@dataclass
class ResultadoEntrenamiento:
    """Resultado de entrenar un candidato con un conjunto de features."""

    nombre_run: str
    nombre_features: str
    tipo: str
    variables: list[str]
    modelo: object
    metricas: dict[str, float]

    @property
    def auc(self) -> float:
        return float(self.metricas["auc"])


def dividir_datos(df: pd.DataFrame, division: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide el dataset en entrenamiento y prueba, estratificando por el objetivo."""
    train, test = train_test_split(
        df,
        test_size=division["test_size"],
        random_state=division["random_state"],
        stratify=df[OBJETIVO],
    )
    return train, test


def preparar_matrices(
    train: pd.DataFrame, test: pd.DataFrame, features: dict
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    """Devuelve X_train, y_train, X_test, y_test y la lista de variables usadas."""
    numericas, categoricas = resolver_features(features["columnas"])
    variables = numericas + categoricas
    return train[variables], train[OBJETIVO], test[variables], test[OBJETIVO], variables


def entrenar_y_evaluar(
    df: pd.DataFrame,
    division: dict,
    features: dict,
    tipo: str,
    parametros: dict,
    umbral: float,
    nombre_run: str = "candidato",
) -> ResultadoEntrenamiento:
    """Entrena un candidato, lo evalúa sobre la parte de prueba y devuelve el resultado."""
    train, test = dividir_datos(df, division)
    X_train, y_train, X_test, y_test, variables = preparar_matrices(train, test, features)
    modelo = construir_modelo(tipo, parametros, *resolver_features(features["columnas"]))
    modelo.fit(X_train, y_train)
    proba = modelo.predict_proba(X_test)[:, 1]
    metricas = calcular_metricas(y_test, proba, umbral)
    return ResultadoEntrenamiento(
        nombre_run=nombre_run,
        nombre_features=features["nombre"],
        tipo=tipo,
        variables=variables,
        modelo=modelo,
        metricas=metricas,
    )


def elegir_mejor(resultados: list[ResultadoEntrenamiento]) -> ResultadoEntrenamiento:
    """Devuelve el candidato con mayor AUC."""
    if not resultados:
        raise ValueError("No hay resultados de entrenamiento para elegir")
    return max(resultados, key=lambda r: r.auc)


def decidir_promocion(auc: float, umbral_auc: float, auc_champion: float | None) -> tuple[bool, str]:
    """Decide si un candidato debe recibir el alias champion.

    Reglas:
    - Debe alcanzar `umbral_auc`.
    - Si ya existe champion, debe superar su AUC (no basta con empatar).
    """
    if auc < umbral_auc:
        return False, f"AUC {auc:.4f} por debajo del mínimo {umbral_auc:.4f}"
    if auc_champion is not None and auc <= auc_champion:
        return False, f"AUC {auc:.4f} no supera al champion ({auc_champion:.4f})"
    if auc_champion is None:
        return True, f"AUC {auc:.4f} supera el mínimo y no hay champion previo"
    return True, f"AUC {auc:.4f} supera al champion ({auc_champion:.4f})"


def resumen_resultados(resultados: list[ResultadoEntrenamiento]) -> list[dict]:
    """Resumen serializable de los resultados, útil para logs y para la memoria."""
    return [
        {
            "nombre_run": r.nombre_run,
            "features": r.nombre_features,
            "tipo": r.tipo,
            "n_features": len(r.variables),
            **r.metricas,
        }
        for r in resultados
    ]
