"""Validación de datos del dataset Consumer Portrait E-Commerce.

Comprueba esquema, nulos, rangos numéricos y categorías permitidas. No depende de
Prefect, MLflow ni de la API: se puede probar con DataFrames en memoria.
"""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from domain.modelo import CATEGORICAS, NUMERICAS, OBJETIVO

# Columnas que se cargan del CSV: las 35 variables de modelado y el objetivo.
COLUMNAS_CSV = NUMERICAS + CATEGORICAS + [OBJETIVO]

# Rango permitido para cada variable numérica. `None` significa sin límite superior.
RANGOS = {
    "Age": (18, 70),
    "Login_Hour": (0, 23),
    "Session_Duration_Min": (2, 180),
    "Pages_Viewed": (1, 100),
    "Search_Count": (0, 60),
    "Wishlist_Count": (0, 30),
    "Cart_Items": (0, 20),
    "Purchase_Count": (0, 10),
    "Purchase_Value": (20, 5000),
    "Discount_Percentage": (0, 80),
    "Product_Price": (10, 5000),
    "Product_Rating": (1, 5),
    "Product_Popularity": (1, 100),
    "Review_Count": (0, 5000),
    "Click_Count": (0, 150),
    "View_Time_Sec": (5, 900),
    "Repeat_Product_View": (0, 20),
    "Browsing_Frequency": (1, 100),
    "Average_Order_Value": (20, 5000),
    "Purchase_Recency_Days": (0, 365),
    "Interaction_Score": (0, 100),
    "Customer_Loyalty_Score": (0, 100),
    "Product_Relevance_Score": (0, 1),
}

CATEGORIAS_VALIDAS = {
    "Gender": {"Female", "Male"},
    "Region": {"North", "Central", "West", "Metro", "South", "Coastal", "Northeast", "East"},
    "Occupation": {
        "Business", "Student", "Retired", "Freelancer", "Government",
        "Doctor", "Homemaker", "Engineer", "Self-Employed", "Teacher",
    },
    "Income_Level": {"Low", "Lower-Middle", "Middle", "Upper-Middle", "High"},
    "Membership_Level": {"Basic", "Silver", "Gold", "Platinum"},
    "Shopping_Device": {"Mobile", "Desktop", "Tablet"},
    "Coupon_Used": {"No", "Yes"},
    "Payment_Method": {
        "Net Banking", "Debit Card", "Cash on Delivery",
        "Digital Wallet", "Credit Card", "UPI",
    },
    "Delivery_Preference": {"Express", "Standard", "Same Day"},
    "Product_Category": {
        "Electronics", "Health", "Home Decor", "Fashion", "Toys", "Furniture",
        "Sports", "Beauty", "Jewelry", "Books", "Automotive", "Grocery",
    },
    "Brand": {
        "Brand_A", "Brand_B", "Brand_C", "Brand_D", "Brand_E",
        "Brand_F", "Brand_G", "Brand_H", "Brand_I", "Brand_J",
    },
    "Favorite_Category": {
        "Electronics", "Health", "Home Decor", "Fashion", "Toys", "Furniture",
        "Sports", "Beauty", "Jewelry", "Books", "Automotive", "Grocery",
    },
    # El CSV trae 5 clases; después de `binarizar_objetivo` el valor es 0/1.
    OBJETIVO: {
        "Recommended", "Neutral", "Not Recommended", "Low Priority", "Highly Recommended",
        0, 1,
    },
}


def validar_esquema(df: pd.DataFrame, columnas: Iterable[str] | None = None) -> None:
    """Falla si falta alguna columna esperada o si hay columnas inesperadas."""
    esperadas = list(columnas or COLUMNAS_CSV)
    faltan = [c for c in esperadas if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en el dataset: {faltan}")
    sobran = [c for c in df.columns if c not in esperadas]
    if sobran:
        raise ValueError(f"Columnas inesperadas en el dataset: {sobran}")


def validar_nulos(df: pd.DataFrame) -> None:
    """Falla si hay valores nulos en alguna columna."""
    nulos = df.isna().sum()
    columnas_con_nulos = [str(c) for c in nulos[nulos > 0].index]
    if columnas_con_nulos:
        raise ValueError(f"Hay valores nulos en: {columnas_con_nulos}")


def validar_rangos(df: pd.DataFrame) -> None:
    """Comprueba rangos numéricos y categorías permitidas en las columnas presentes."""
    for columna, (minimo, maximo) in RANGOS.items():
        if columna not in df.columns:
            continue
        valores = pd.to_numeric(df[columna], errors="coerce")
        if valores.isna().any():
            raise ValueError(f"La columna '{columna}' tiene valores no numéricos")
        if minimo is not None and (valores < minimo).any():
            raise ValueError(f"La columna '{columna}' tiene valores menores que {minimo}")
        if maximo is not None and (valores > maximo).any():
            raise ValueError(f"La columna '{columna}' tiene valores mayores que {maximo}")

    for columna, permitidas in CATEGORIAS_VALIDAS.items():
        if columna not in df.columns:
            continue
        desconocidas = set(df[columna].dropna().unique()) - permitidas
        if desconocidas:
            raise ValueError(f"La columna '{columna}' tiene categorías no permitidas: {sorted(desconocidas)}")


def validar_datos(df: pd.DataFrame, columnas: Iterable[str] | None = None) -> pd.DataFrame:
    """Ejecuta todas las validaciones y devuelve el DataFrame para encadenar."""
    validar_esquema(df, columnas)
    validar_nulos(df)
    validar_rangos(df)
    return df


def validar_features(df: pd.DataFrame) -> None:
    """Valida solo las variables de entrada; admite columnas extra como `prediccion`."""
    columnas = [c for c in COLUMNAS_CSV if c != OBJETIVO]
    faltan = [c for c in columnas if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en el dataset: {faltan}")
    validar_nulos(df[columnas])
    validar_rangos(df[columnas])
