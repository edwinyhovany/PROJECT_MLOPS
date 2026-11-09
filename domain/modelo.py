"""Lógica de negocio del modelo de recomendación E-Commerce: variables, pipeline y métricas.

Dataset: Consumer Portrait E-Commerce Recommendation Dataset
Objetivo: Predecir si un producto será 'Recommended' para un cliente (clasificación binaria).
La variable objetivo 'Recommendation_Label' se binariza: 'Recommended' → 1, el resto → 0.

No usa Prefect ni MLflow: se puede probar sin servicios levantados.
"""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Variables del modelo. Los identificadores (Customer_ID, Session_ID, Product_ID) no se usan.
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
VARIABLES = NUMERICAS + CATEGORICAS
OBJETIVO = "Recommendation_Label"
ETIQUETA_POSITIVA = "Recommended"


def binarizar_objetivo(serie):
    """Convierte Recommendation_Label a binario: 'Recommended' → 1, el resto → 0."""
    return (serie == ETIQUETA_POSITIVA).astype(int)


def resolver_features(columnas):
    """Convierte `columnas` de config.yaml ('ALL' o una lista de nombres) en (numericas, categoricas)."""
    if columnas == "ALL":
        elegidas = VARIABLES
    else:
        desconocidas = [v for v in columnas if v not in VARIABLES]
        if desconocidas:
            raise ValueError(f"variables desconocidas en 'columnas': {desconocidas}")
        elegidas = columnas

    if not elegidas:
        raise ValueError("la selección de features no tiene ninguna variable")
    numericas = [v for v in NUMERICAS if v in elegidas]
    categoricas = [v for v in CATEGORICAS if v in elegidas]
    return numericas, categoricas


def construir_modelo(tipo, parametros, numericas, categoricas):
    """Preprocesado (escalar y one-hot) más el estimador elegido en la configuración."""
    if tipo == "logistic_regression":
        estimador = LogisticRegression(**parametros)
    elif tipo == "gradient_boosting":
        estimador = GradientBoostingClassifier(**parametros)
    else:
        raise ValueError(f"modelo '{tipo}' no soportado")

    transformadores = []
    if numericas:
        transformadores.append(("num", StandardScaler(), numericas))
    if categoricas:
        transformadores.append(("cat", OneHotEncoder(handle_unknown="ignore"), categoricas))
    preprocesado = ColumnTransformer(transformadores)
    return Pipeline([("preprocesado", preprocesado), ("modelo", estimador)])


def calcular_metricas(y_true, proba, umbral=0.5):
    """Métricas de clasificación binaria a partir de la probabilidad de recomendación."""
    pred = (proba >= umbral).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }
