"""API FastAPI que sirve el modelo champion.

Carga el modelo registrado con alias 'champion' y expone un endpoint
para predecir churn dado los datos de un cliente.
"""

import os
import sys
import uvicorn

# Añadir el directorio raíz al path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow

# Configuración
app = FastAPI(
    title="API de Predicción Churn",
    description="Servicio de inferencia que sirve el modelo champion",
    version="1.0.0",
)

# Modelos de petición y respuesta
class ClienteDatos(BaseModel):
    Age: float
    Gender: str
    Region: str
    Occupation: str
    Income_Level: str
    Membership_Level: str
    Shopping_Device: str
    Coupon_Used: str
    Payment_Method: str
    Delivery_Preference: str
    Product_Category: str
    Brand: str
    Favorite_Category: str
    # Variables numéricas adicionales
    Login_Hour: float = 0.0
    Session_Duration_Min: float = 0.0
    Pages_Viewed: float = 0.0
    Search_Count: float = 0.0
    Wishlist_Count: float = 0.0
    Cart_Items: float = 0.0
    Purchase_Count: float = 0.0
    Purchase_Value: float = 0.0
    Discount_Percentage: float = 0.0
    Product_Price: float = 0.0
    Product_Rating: float = 0.0
    Product_Popularity: float = 0.0
    Review_Count: float = 0.0
    Click_Count: float = 0.0
    View_Time_Sec: float = 0.0
    Repeat_Product_View: float = 0.0
    Browsing_Frequency: float = 0.0
    Average_Order_Value: float = 0.0
    Purchase_Recency_Days: float = 0.0
    Interaction_Score: float = 0.0
    Customer_Loyalty_Score: float = 0.0
    Product_Relevance_Score: float = 0.0

class ResultadoPrediccion(BaseModel):
    probabilidad_churn: float
    predicion_churn: int  # 1 = churn, 0 = no churn

# Cargar el modelo champion al iniciar
try:
    model = mlflow.pyfunc.load_model(model_uri="champion")
    print("✅ Modelo champion cargado correctamente")
except Exception as e:
    print(f"⚠️  No se pudo cargar el modelo champion: {e}")
    model = None


@app.get("/")
def root():
    """Endpoint raíz con información de la API."""
    return {
        "mensaje": "API de Predicción Churn",
        "modelo": "champion",
        "estado": "listo" if model else "error_cargando_modelo"
    }


@app.post("/predecir", response_model=ResultadoPrediccion)
def predecir(cliente: ClienteDatos):
    """Realiza una predicción de churn para un cliente."""
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")
    
    # Preparar datos para la predicción
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
    
    import pandas as pd
    datos_dict = cliente.model_dump()
    row = {}
    for var in numericas:
        row[var] = datos_dict.get(var, 0.0)
    for var in categoricas:
        row[var] = datos_dict.get(var, "desconocido")
    
    df = pd.DataFrame([row])
    
    # Realizar predicción
    proba = model.predict_proba(df)[:, 1]
    pred = int(proba >= 0.5)
    
    return {
        "probabilidad_churn": float(proba[0]),
        "predicion_churn": pred,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)