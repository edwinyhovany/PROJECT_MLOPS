"""Carga el dataset de Consumer Portrait E-Commerce, valida sus columnas y limpia nulos.

Dataset: https://www.kaggle.com/datasets/colabsss/consumer-portrait-e-commerce-dataset
Por defecto lee `datos.ruta` de configs/config.yaml. Las rutas se resuelven desde el
directorio de trabajo, así que ejecuta el script desde la raíz del proyecto.
"""
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

RUTA_CONFIG = "configs/config.yaml"

# Columnas numéricas del modelo
NUMERICAS = [
    "Age",
    "Login_Hour",
    "Session_Duration_Min",
    "Pages_Viewed",
    "Search_Count",
    "Wishlist_Count",
    "Cart_Items",
    "Purchase_Count",
    "Purchase_Value",
    "Discount_Percentage",
    "Product_Price",
    "Product_Rating",
    "Product_Popularity",
    "Review_Count",
    "Click_Count",
    "View_Time_Sec",
    "Repeat_Product_View",
    "Browsing_Frequency",
    "Average_Order_Value",
    "Purchase_Recency_Days",
    "Interaction_Score",
    "Customer_Loyalty_Score",
    "Product_Relevance_Score",
]

# Columnas categóricas del modelo
CATEGORICAS = [
    "Gender",
    "Region",
    "Occupation",
    "Income_Level",
    "Membership_Level",
    "Shopping_Device",
    "Coupon_Used",
    "Payment_Method",
    "Delivery_Preference",
    "Product_Category",
    "Brand",
    "Favorite_Category",
]

OBJETIVO = "Recommendation_Label"
COLUMNAS = NUMERICAS + CATEGORICAS + [OBJETIVO]


def cargar_datos(ruta=None):
    """Devuelve el DataFrame de Consumer Portrait con validación y tratamiento de nulos.
    
    - Valida existencia del archivo y presencia de columnas esperadas.
    - Imputa nulos si llegasen a existir (mediana en numéricas, 'Desconocido'/'Moda' en categóricas).
    - Convierte categóricas a string de manera limpia.
    - Elimina registros con objetivo nulo si los hubiera.
    """
    if ruta is None:
        config_path = Path(RUTA_CONFIG)
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                ruta = cfg.get("datos", {}).get("ruta", "data/Consumer_Portrait_ECommerce_Recommendation_Dataset.csv")
        else:
            ruta = "data/Consumer_Portrait_ECommerce_Recommendation_Dataset.csv"

    path = Path(ruta)
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra el dataset: {path}")

    df = pd.read_csv(path)
    faltan = [c for c in COLUMNAS if c not in df.columns]
    if faltan:
        raise ValueError(f"Al CSV le faltan columnas requeridas: {faltan}")

    df = df[COLUMNAS].copy()

    # 1. Tratamiento y validación contra valores nulos (NaN / None)
    # Si la variable objetivo tiene nulos, descartamos esas filas
    if df[OBJETIVO].isna().any():
        df = df.dropna(subset=[OBJETIVO])

    # Imputar numéricas con la mediana o 0 si todo fuera NaN
    for col in NUMERICAS:
        if df[col].isna().any():
            mediana = df[col].median()
            df[col] = df[col].fillna(mediana if not pd.isna(mediana) else 0.0)
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Imputar categóricas con valor seguro y castear a string
    for col in CATEGORICAS:
        if df[col].isna().any():
            moda = df[col].mode()
            val_imputar = moda[0] if not moda.empty else "Desconocido"
            df[col] = df[col].fillna(val_imputar)
        df[col] = df[col].astype(str)

    return df


def main():
    ruta = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        df = cargar_datos(ruta)
        print(f"Filas: {len(df)}  columnas: {df.shape[1]}  nulos totales: {int(df.isna().sum().sum())}")
        print(f"Distribución de {OBJETIVO}:")
        print(df[OBJETIVO].value_counts())
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
