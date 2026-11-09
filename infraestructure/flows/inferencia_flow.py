"""Pipeline de inferencia en Prefect.

Carga el modelo por su alias champion, calcula predicciones sobre un lote de datos
y guarda las predicciones en un archivo con la fecha de ejecución.
"""

import os
import sys

# Añadir el directorio raíz (3 niveles hacia arriba desde flows/) al path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Ahora añadir scripts al path (4 niveles hacia arriba + scripts/)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts"))

import mlflow
import pandas as pd
from prefect import flow, task


@task
def cargar_datos_prueba():
    """Carga unos datos de prueba para simular clientes nuevos.

    En producción esto recibiría un lote real. Aquí usamos el dataset de validación
    o generamos una muestra aleatoria.
    """
    from scripts.load_data import cargar_datos
    df = cargar_datos()
    # Tomar una muestra aleatoria de 5 registros para simular clientes nuevos
    return df.sample(5, random_state=42)


@task
def predecir_con_champion(df):
    """Carga el modelo por su alias champion y hace predicciones.

    El alias 'champion' se resuelve automáticamente hacia la versión actual
    mejor registrada.
    """
    import mlflow

    # Cargar el modelo por alias champion (no por versión fija)
    model = mlflow.pyfunc.load_model(model_uri="champion")

    # Realizar predicciones
    # Usar solo features numéricas y categóricas definidas en el dominio
    numericas = [
        "Age", "Login_Hour", "Session_Duration_Min", "Pages_Viewed",
        "Search_Count", "Wishlist_Count", "Cart_Items", "Purchase_Count",
        "Purchase_Value", "Discount_Percentage", "Product_Price",
        "Product_Rating", "Product_Popularity", "Review_Count",
        "Click_Count", "View_Time_Sec", "Repeat_Product_View",
        "Browsing_Frequency", "Average_Order_Value",
        "Purchase_Recency_Days", "Interaction_Score",
        "Customer_Loyalty_Score", "Product_Relevance_Score",
    ]
    categoricas = [
        "Gender", "Region", "Occupation", "Income_Level",
        "Membership_Level", "Shopping_Device", "Coupon_Used",
        "Payment_Method", "Delivery_Preference", "Product_Category",
        "Brand", "Favorite_Category",
    ]

    X = df[numericas + categoricas]
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    resultados = df.copy()
    resultados["probabilidad_churn"] = proba
    resultados["prediccion_churn"] = pred
    return resultados


@task
def guardar_predicciones(resultados, filename=None):
    """Guarda las predicciones en un archivo con la fecha de ejecución."""
    if filename is None:
        fecha = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"predicciones_{fecha}.csv"

    resultados.to_csv(filename, index=False)
    return filename


@flow(name="inferencia-champion-v1")
def inferencia():
    """Flow principal de inferencia con horario diario.

    Carga el modelo champion, hace predicciones y guarda resultados.
    """
    # 1. Cargar datos de prueba
    df = cargar_datos_prueba()
    print(f"Datos de prueba cargados: {df.shape[0]} registros")

    # 2. Predecir con el modelo champion
    resultados = predecir_con_champion(df)
    print(f"Predicciones generadas: {len(resultados)} registros")

    # 3. Guardar predicciones con fecha
    filename = guardar_predicciones(resultados)
    print(f"Predicciones guardadas en: {filename}")

    return filename


if __name__ == "__main__":
    import datetime
    os.environ["PREFECT_API_URL"] = os.environ.get("PREFECT_API_URL", "http://localhost:4200/api")
    os.environ["MLFLOW_TRACKING_URI"] = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    inferencia()