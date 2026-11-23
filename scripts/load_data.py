"""Carga el dataset Consumer Portrait E-Commerce, valida sus columnas y muestra su tamaño.

Las rutas se resuelven desde la raíz del proyecto, no desde el directorio de trabajo, para
que los flows, la API y los tests funcionen igual desde cualquier carpeta.

Por defecto el objetivo `Recommendation_Label` se devuelve binarizado (`Recommended` = 1,
resto = 0), que es la variable que usan el entrenamiento, la inferencia y el drift. Para
inspeccionar las 5 clases originales usa `cargar_datos(binarizar=False)`.

Ejecución (desde cualquier carpeta):
    poetry run python scripts/load_data.py

Para usarla desde un flow:
    from scripts.load_data import cargar_datos
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from domain.modelo import OBJETIVO, binarizar_objetivo
from domain.validacion import COLUMNAS_CSV, validar_datos

ROOT_DIR = Path(__file__).resolve().parent.parent
RUTA_CONFIG = ROOT_DIR / "configs" / "config.yaml"

# Columnas del dataset: las 35 variables predictivas y el objetivo.
# Los identificadores (Customer_ID, Session_ID, Product_ID) y Consumer_Portrait_Segment no se usan.
COLUMNAS = COLUMNAS_CSV


def resolver_ruta(ruta=None):
    """Devuelve la ruta resuelta a partir de config.yaml o del argumento suministrado."""
    if ruta is None:
        if RUTA_CONFIG.exists():
            with open(RUTA_CONFIG, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                ruta = cfg.get("datos", {}).get("ruta", "data/Consumer_Portrait_ECommerce_Recommendation_Dataset.csv")
        else:
            ruta = "data/Consumer_Portrait_ECommerce_Recommendation_Dataset.csv"

    p = Path(ruta)
    if not p.is_absolute() and not p.exists():
        p = (ROOT_DIR / p).resolve()
    return p


def cargar_config(ruta=None):
    """Lee el YAML de configuración (por defecto `configs/config.yaml`)."""
    ruta = Path(ruta) if ruta is not None else RUTA_CONFIG
    if not ruta.is_absolute():
        ruta = ROOT_DIR / ruta
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def cargar_datos(ruta=None, validar=True, binarizar=True):
    """Devuelve el DataFrame de Consumer Portrait con las columnas esperadas.

    - `validar=True`: comprueba esquema, nulos, rangos y categorías antes de binarizar.
    - `binarizar=True`: convierte `Recommendation_Label` a 0/1 (Recommended = 1).
    """
    path = resolver_ruta(ruta)
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra el dataset en: {path}")

    df = pd.read_csv(path)
    faltan = [c for c in COLUMNAS if c not in df.columns]
    if faltan:
        raise ValueError(f"Al CSV le faltan columnas: {faltan}")

    df = df[COLUMNAS].copy()
    if validar:
        validar_datos(df)
    if binarizar:
        df[OBJETIVO] = binarizar_objetivo(df[OBJETIVO])
    return df


def main():
    df_crudo = cargar_datos(binarizar=False)
    df = cargar_datos()
    print(f"Filas: {len(df)}  columnas: {df.shape[1]}  nulos: {int(df.isna().sum().sum())}")
    print(f"Columnas: {list(df.columns)}")
    print("\nDistribución original de Recommendation_Label:")
    print(df_crudo[OBJETIVO].value_counts())
    print("\nDistribución binaria usada por el modelo:")
    print(df[OBJETIVO].value_counts())
    print(f"Tasa positiva: {df[OBJETIVO].mean():.2%}")


if __name__ == "__main__":
    main()
