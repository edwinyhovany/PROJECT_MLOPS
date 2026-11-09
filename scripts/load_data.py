"""Carga el dataset de Consumer Portrait E-Commerce, valida sus columnas y muestra su tamaño.

Por defecto lee `datos.ruta` de configs/config.yaml. Las rutas se resuelven desde la raíz
del proyecto o desde el directorio de trabajo actual.

Ejecución (desde la raíz del proyecto):
    poetry run python scripts/load_data.py

Para usarla desde un flow:
    from scripts.load_data import cargar_datos
"""
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

# Raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent
RUTA_CONFIG = ROOT_DIR / "configs" / "config.yaml"

# Columnas del dataset de Consumer Portrait: variables predictivas y el objetivo `Recommendation_Label`.
# Los identificadores (Customer_ID, Session_ID, Product_ID) no se usan como features.
COLUMNAS = [
    # Numéricas
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
    # Categóricas
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
    # Objetivo
    "Recommendation_Label",
]


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


def cargar_datos(ruta=None):
    """Devuelve el DataFrame de Consumer Portrait con las columnas esperadas. Falla si falta alguna."""
    path = resolver_ruta(ruta)
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra el dataset en: {path}")

    df = pd.read_csv(path)
    faltan = [c for c in COLUMNAS if c not in df.columns]
    if faltan:
        raise ValueError(f"Al CSV le faltan columnas: {faltan}")

    # Tratamiento seguro contra nulos
    df_clean = df[COLUMNAS].copy()
    if df_clean["Recommendation_Label"].isna().any():
        df_clean = df_clean.dropna(subset=["Recommendation_Label"])

    return df_clean


def main():
    ruta = sys.argv[1] if len(sys.argv) > 1 else None  # None = datos.ruta de configs/config.yaml
    try:
        df = cargar_datos(ruta)
        print(f"Filas: {len(df)}  columnas: {df.shape[1]}  nulos: {int(df.isna().sum().sum())}")
        print(f"Columnas: {list(df.columns)}")
        print(f"\nDistribución de Recommendation_Label:")
        print(df["Recommendation_Label"].value_counts())
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
