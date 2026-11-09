# Proyecto MLOps: Consumer Portrait E-Commerce Recommendation

Este repositorio implementa el ciclo de vida completo de un proyecto MLOps para la **Actividad 2** (construido sobre los fundamentos de la Actividad 1):
1. **Pipeline de entrenamiento** orquestada con **Prefect**, evaluando modelos y registrándolos en **MLflow Tracking**.
2. **Registro de modelos y estrategia Champion**: el modelo solo se promociona a champion si supera el umbral de AUC y al champion vigente.
3. **Pipeline de inferencia diaria** en Prefect que consume el modelo champion y persiste predicciones por lotes.
4. **API REST de servicio con FastAPI** que expone el modelo champion para inferencia en tiempo real.
5. **Configuración desacoplada en YAML** (`configs/config.yaml` y `configs/features.yaml`).

---

## Dataset del Proyecto

- **Origen del dataset:** Descargado directamente de Kaggle:  
  [Consumer Portrait E-Commerce Recommendation Dataset (Kaggle)](https://www.kaggle.com/datasets/colabsss/consumer-portrait-e-commerce-dataset)
- **Ubicación local en el repositorio:**  
  [`data/Consumer_Portrait_ECommerce_Recommendation_Dataset.csv`](data/Consumer_Portrait_ECommerce_Recommendation_Dataset.csv)
- **Dimensión:** 15,000 filas y 40 columnas (comportamiento de navegación, demografía, compras y producto).
- **Variable objetivo (`Recommendation_Label`):** Predicción de recomendación binarizada (`Recommended` = 1, `Neutral`/`Not Recommended`/`Low Priority` = 0).
- **Destino de las predicciones generadas:**  
  Las predicciones por lotes generadas por la pipeline de inferencia diaria se guardan automáticamente en la carpeta [`data/`](data/) con el formato:
  ```text
  data/prediccion_YYYY-MM-DD.csv
  ```
  *(Por ejemplo: `data/prediccion_2026-11-09.csv`). Cada archivo contiene las variables del lote de clientes evaluado junto con `probabilidad_recomendacion` y `prediccion_recomendado`.*

---

## Evidencias de Ejecución (Carpeta docs)

En la carpeta [`docs/`](docs/) se encuentran los soportes y pantallazos de la ejecución completa de ambas actividades en formato PDF:

- [`docs/Actividad 1_ Fundamentos, versionamiento y tracking.pdf`](docs/Actividad%201_%20Fundamentos,%20versionamiento%20y%20tracking.pdf): Evidencias de entorno Poetry, servicios Docker de MLflow y Prefect levantados, y validación del primer flow.
- [`docs/Actividad 2_Pipeline y empaquetado.pdf`](docs/Actividad%202_Pipeline%20y%20empaquetado.pdf): Evidencias del pipeline de entrenamiento, registro de runs y métricas en MLflow, promoción a champion en Model Registry, pipeline de inferencia diaria, despliegues en Prefect y consumo de la API FastAPI.

---

## Estructura del Proyecto

```text
PROJECT_MLOPS/
├── configs/
│   ├── config.yaml                     # Configuración de datos, división, modelos, MLflow, registro e inferencia
│   └── features.yaml                   # Selección de features del experimento (ALL, NUMERIC, CUSTOM)
├── data/
│   ├── Consumer_Portrait_ECommerce_Recommendation_Dataset.csv   # Dataset real de Kaggle
│   └── prediccion_*.csv                # Archivos generados por el pipeline de inferencia diaria
├── docs/
│   ├── Actividad 1_ Fundamentos, versionamiento y tracking.pdf  # Evidencias Actividad 1
│   └── Actividad 2_Pipeline y empaquetado.pdf                   # Evidencias Actividad 2
├── domain/
│   ├── __init__.py
│   └── modelo.py                       # Lógica de negocio pura (features, pipeline scikit-learn, métricas)
├── infraestructure/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py                     # API FastAPI que sirve el modelo champion
│   └── flows/
│       ├── __init__.py
│       ├── entrenamiento_flow.py        # Pipeline de entrenamiento y registro en Prefect/MLflow
│       ├── inferencia_flow.py           # Pipeline de inferencia en lote en Prefect
│       └── desplegar.py                 # Script de despliegue de ambos flows en Prefect
├── scripts/
│   ├── __init__.py
│   └── load_data.py                    # Carga y validación defensiva contra valores nulos
├── services/
│   ├── docker-compose.yml              # Levanta contenedores de MLflow (5000) y Prefect (4200)
│   └── requirements.txt
├── pyproject.toml                      # Gestión de dependencias con Poetry
├── poetry.lock
└── README.md
```

---

## Requisitos y Entorno

- **Python:** 3.12 a 3.14
- **Poetry:** 2.x
- **Docker Desktop:** Para levantar los servidores de MLflow y Prefect

### Instalación de dependencias

```powershell
poetry install
```

### Variables de entorno requeridas

En cada terminal donde vayas a ejecutar un flow o la API:

```powershell
$env:PREFECT_API_URL = "http://localhost:4200/api"
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:PYTHONUTF8 = "1"
```

---

## Guía de Ejecución Paso a Paso

### Paso 1: Levantar los Servicios (MLflow y Prefect)

Desde la raíz del proyecto:

```powershell
cd services
docker compose up -d
cd ..
```

Comprobar que responden:
- **MLflow UI:** [http://localhost:5000](http://localhost:5000)
- **Prefect UI:** [http://localhost:4200](http://localhost:4200)

---

### Paso 2: Pipeline de Entrenamiento

Ejecuta el pipeline de entrenamiento en Prefect:

```powershell
poetry run python infraestructure\flows\entrenamiento_flow.py
```

**Qué realiza:**
1. **Ingesta:** Carga `data/Consumer_Portrait_ECommerce_Recommendation_Dataset.csv` y aplica tratamiento de nulos.
2. **Entrenamiento:** Entrena los candidatos configurados (`ecommerce-logreg`, `ecommerce-gbt`) usando el subconjunto de features definido en `configs/features.yaml`.
3. **MLflow Tracking:** Crea runs con el tag `fase = entrenamiento`, registrando parámetros, métricas (`auc`, `accuracy`, `precision`, `recall`, `f1`), tags `features` y `n_features`, y el artefacto del modelo.
4. **Champion Promotion:** Compara el mejor modelo contra el champion vigente. Solo si supera el umbral mínimo (`registro.umbral_auc = 0.60`) y mejora el AUC del champion actual, se promueve al alias **`champion`** en el Model Registry de MLflow.
5. **Prefect Artifact:** Genera un resumen visual en Markdown dentro del run de Prefect.

---

### Paso 3: Pipeline de Inferencia

Ejecuta el pipeline de inferencia por lotes:

```powershell
poetry run python infraestructure\flows\inferencia_flow.py
```

**Qué realiza:**
1. Carga el modelo desde MLflow utilizando el alias dinámico `models:/ecommerce-recommendation-modelo@champion`.
2. Simula un lote diario de clientes desde el dataset.
3. Genera probabilidades y clasificaciones.
4. **Guarda el CSV** con la fecha del día en [`data/prediccion_YYYY-MM-DD.csv`](data/).
5. Registra un run en MLflow con tag `fase = inferencia`, registrando métricas de lote y adjuntando el CSV como artefacto.

---

### Paso 4: Despliegues en Prefect

Para registrar ambos flujos como despliegues en Prefect:

```powershell
poetry run python infraestructure\flows\desplegar.py
```

- **`entrenamiento-ecommerce`:** Despliegue manual (para ejecutar bajo demanda).
- **`inferencia-ecommerce-diaria`:** Despliegue con horario diario programado (cron `0 6 * * *` a las 06:00 UTC).

---

### Paso 5: Servir el Modelo Champion con FastAPI

Inicia el servidor de inferencia en tiempo real:

```powershell
poetry run uvicorn infraestructure.api.main:app --port 8000
```

- **Documentación Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check y Versión Champion:** [http://localhost:8000/health](http://localhost:8000/health)
- **Inspección de Modelos:** [http://localhost:8000/v1/models](http://localhost:8000/v1/models)

#### Ejemplo de predicción (`POST /predict`):

Desde otra terminal PowerShell:

```powershell
$cliente = @{
    Age = 33.0
    Login_Hour = 1.0
    Session_Duration_Min = 150.0
    Pages_Viewed = 80.0
    Search_Count = 5.0
    Wishlist_Count = 10.0
    Cart_Items = 4.0
    Purchase_Count = 3.0
    Purchase_Value = 2500.0
    Discount_Percentage = 20.0
    Product_Price = 1500.0
    Product_Rating = 4.0
    Product_Popularity = 50.0
    Review_Count = 500.0
    Click_Count = 40.0
    View_Time_Sec = 120.0
    Repeat_Product_View = 2.0
    Browsing_Frequency = 15.0
    Average_Order_Value = 2800.0
    Purchase_Recency_Days = 10.0
    Interaction_Score = 85.0
    Customer_Loyalty_Score = 60.0
    Product_Relevance_Score = 0.85
    Gender = "Male"
    Region = "North"
    Occupation = "Business"
    Income_Level = "Middle"
    Membership_Level = "Silver"
    Shopping_Device = "Mobile"
    Coupon_Used = "No"
    Payment_Method = "Credit Card"
    Delivery_Preference = "Express"
    Product_Category = "Electronics"
    Brand = "Brand_A"
    Favorite_Category = "Electronics"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body $cliente
```

**Respuesta recibida:**
```json
{
  "probabilidad_recomendacion": 0.8412,
  "prediccion": 1,
  "etiqueta": "Recommended",
  "version_modelo": "1"
}
```

---

## Criterio de Éxito de la Actividad

1. Al ejecutar un reentrenamiento que supere las métricas del champion actual, MLflow traslada automáticamente el alias `champion` hacia la nueva versión.
2. Al reiniciar la API (`uvicorn`), esta sirve inmediatamente la nueva versión del modelo sin modificar una sola línea de código en la API.
