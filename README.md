# Proyecto MLOps: Consumer Portrait E-Commerce Recommendation

Este repositorio implementa el ciclo de vida integral de un proyecto MLOps para la **Actividad 3** (construido y evolucionado sobre los fundamentos de las Actividades 1 y 2):
1. **Pipeline de entrenamiento y registro continuo**: Orquestado con **Prefect**, evaluando múltiples familias de modelos (`LogisticRegression`, `HistGradientBoostingClassifier`) y combinaciones de variables (`all`, `custom`), registrando métricas y artefactos en **MLflow Tracking**.
2. **Registro de modelos y estrategia Champion**: Promoción automática basada en umbral estricto de AUC contra el champion vigente en el MLflow Model Registry.
3. **Pipeline de inferencia con emulación de drift**: Ingesta de lotes de prueba, inyección configurable de drift sintético (`0.0`, `0.3`, `0.6`), generación de predicciones y registro de trazabilidad en MLflow.
4. **Monitoreo continuo de Data Drift (Features Drift)**: Detección estadística mediante **Evidently** generando reportes HTML y alertas automáticas basadas en umbral configurable (`monitoreo.umbral_drift`).
5. **Monitoreo continuo de Concept / Event Drift (Target Drift)**: Prueba estadística de bondad de ajuste chi-cuadrado para contrastar la distribución observada contra el entrenamiento de referencia, validando y rechazando lotes sin etiquetas.
6. **API REST de servicio con FastAPI**: Exposición en tiempo real del modelo champion activo sin acoplamiento a versiones estáticas.
7. **Suite de pruebas y CI/CD**: 117 tests automatizados con `pytest` que validan contratos de datos, modelos, inferencia, API y lógica de monitoreo sin depender de servicios externos en ejecución.

---

## Dataset del Proyecto

- **Origen del dataset:** Descargado directamente de Kaggle:  
  [Consumer Portrait E-Commerce Recommendation Dataset (Kaggle)](https://www.kaggle.com/datasets/colabsss/consumer-portrait-e-commerce-dataset)
- **Ubicación local en el repositorio:**  
  [`data/Consumer_Portrait_ECommerce_Recommendation_Dataset.csv`](data/Consumer_Portrait_ECommerce_Recommendation_Dataset.csv)
- **Dimensión:** 15,000 filas y 40 columnas (comportamiento de navegación, demografía, compras y características del producto).
- **Variable objetivo (`Recommendation_Label`):** Predicción de recomendación binarizada (`Recommended` = 1, `Neutral`/`Not Recommended`/`Low Priority` = 0).
- **Destino de las predicciones generadas:**  
  Las predicciones por lotes generadas por la pipeline de inferencia diaria se guardan automáticamente en la carpeta [`data/`](data/) con el formato:
  ```text
  data/prediccion.csv
  data/prediccion_evidencia_nivel_X.X.csv
  ```
  *(Por ejemplo: `data/prediccion.csv`). Cada archivo contiene las variables del lote evaluado junto con `probabilidad_recomendacion` y `prediccion`.*

---

## Evidencias de Ejecución (Carpeta docs)

En la carpeta [`docs/`](docs/) se encuentran los soportes y evidencias de ejecución en formato PDF:

- [`docs/Actividad 1_ Fundamentos, versionamiento y tracking.pdf`](docs/Actividad%201_%20Fundamentos,%20versionamiento%20y%20tracking.pdf): Evidencias de entorno Poetry, servicios Docker de MLflow y Prefect levantados, y validación del primer flow.
- [`docs/Actividad 2_Pipeline y empaquetado.pdf`](docs/Actividad%202_Pipeline%20y%20empaquetado.pdf): Evidencias del pipeline de entrenamiento, registro de runs y métricas en MLflow, promoción a champion en Model Registry, pipeline de inferencia diaria, despliegues en Prefect y consumo de la API FastAPI.
- [`docs/Actividad 3_Monitoreo, drift y CI.pdf`](docs/Actividad%203_Monitoreo,%20drift%20y%20CI.pdf): Evidencias del monitoreo continuo del modelo, detección de *data drift* y *concept drift* (con Evidently AI), alertas automatizadas, reentrenamiento disparado por umbral y flujo de integración continua (CI) en GitHub Actions.

---

## Estructura del Proyecto

```text
PROJECT_MLOPS/
├── .github/
│   └── workflows/
│       └── ci.yml                      # Pipeline de integración continua (pruebas, evidencia de drift y artefactos)
├── configs/
│   └── config.yaml                     # Configuración unificada (datos, features, candidatos, MLflow, inferencia y monitoreo)
├── data/
│   ├── Consumer_Portrait_ECommerce_Recommendation_Dataset.csv   # Dataset real de Kaggle
│   └── prediccion_*.csv                # Predicciones generadas por el pipeline de inferencia
├── docs/
│   ├── Actividad 1_ Fundamentos, versionamiento y tracking.pdf  # Evidencias Actividad 1
│   ├── Actividad 2_Pipeline y empaquetado.pdf                   # Evidencias Actividad 2
│   └── Actividad 3: Monitoreo, drift y CI.pdf                   # Evidencias Actividad 3
├── domain/
│   ├── modelo.py                       # Definición de variables numéricas, categóricas y pipelines scikit-learn
│   ├── validacion.py                   # Validación de rangos, categorías, nulos y tipos
│   ├── entrenamiento.py                # Lógica pura de partición, entrenamiento y regla de promoción champion
│   ├── inferencia.py                   # Muestreo de lotes, inyección de drift (0.0 a 0.6) y guardado histórico
│   └── monitoreo.py                   # Monitoreo de features con Evidently y drift de target con test chi-cuadrado
├── infraestructure/
│   ├── api/
│   │   └── main.py                     # API FastAPI que sirve el modelo champion
│   └── flows/
│       ├── entrenamiento_flow.py        # Pipeline de entrenamiento y registro en Prefect/MLflow
│       ├── inferencia_flow.py           # Pipeline de inferencia diaria con drift emulado
│       ├── drift_features_flow.py       # Pipeline de monitoreo de drift en features
│       ├── drift_evento_flow.py         # Pipeline de monitoreo de drift en target (evento)
│       └── desplegar.py                 # Orquestación de despliegues programados en Prefect
├── lotes/                              # Lotes de prueba guardados para auditoría (lote_nivel_0.0.csv, lote_actual.csv, etc.)
├── reports/                            # Reportes de monitoreo (drift_features_*.html, drift_evento_*.json, evidencia_drift.json)
├── scripts/
│   ├── load_data.py                    # Carga y validación desacoplada con resolución automática de ruta
│   └── evidencia_drift.py              # Script ejecutable para generar la comparación reproducible entre niveles de drift
├── services/
│   ├── docker-compose.yml              # Levanta contenedores de MLflow (puerto 5000) y Prefect (puerto 4200)
│   └── requirements.txt
├── tests/                              # Suite de 117 pruebas unitarias y de integración
│   ├── test_api.py
│   ├── test_datos.py
│   ├── test_drift_flows.py
│   ├── test_entrenamiento.py
│   ├── test_inferencia.py
│   ├── test_modelo.py
│   └── test_monitoreo.py
├── pyproject.toml                      # Gestión de dependencias con Poetry
├── poetry.lock
└── README.md
```

---

## Requisitos y Entorno

- **Python:** 3.12
- **Poetry:** 2.x
- **Docker Desktop:** Para levantar los servidores de MLflow y Prefect

### Instalación de dependencias

```powershell
poetry install
```

### Variables de entorno

```powershell
$env:PREFECT_API_URL = "http://localhost:4200/api"
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:PYTHONUTF8 = "1"
```

*Nota: Los flows cuentan con valores por defecto hacia `http://localhost:4200/api` y `http://localhost:5000` si estas variables no fueron definidas en la sesión.*

---

## Configuración Unificada (`configs/config.yaml`)

Toda la configuración del ciclo de vida se encuentra centralizada en [`configs/config.yaml`](configs/config.yaml):

- **`datos.ruta`**: Ruta del CSV de entrada (`data/Consumer_Portrait_ECommerce_Recommendation_Dataset.csv`).
- **`division`**: Partición reproducible (`test_size: 0.2`, `random_state: 42`).
- **`features`**: Grupos de variables experimentales (`all` con 35 features, y `custom` con 15 variables de alto impacto).
- **`candidatos`**: Modelos candidatos a evaluar (`ecommerce-logreg-c1`, `ecommerce-logreg-c01`, `ecommerce-gbt`).
- **`mlflow`**: Tracking URI, experimento (`ecommerce-recommendation-mlops`) y prefijos.
- **`metricas.umbral`**: Umbral de corte de clasificación (0.5).
- **`registro`**: Nombre en el Model Registry (`ecommerce-recommendation-modelo`) y umbral mínimo de AUC (0.60).
- **`inferencia`**: Tamaño de lote diario (100 filas) y nivel de drift a emular (`nivel_drift: 0.0`).
- **`monitoreo`**:
  - `carpeta_reportes`: Directorio de reportes generados (`reports`).
  - `lotes_dir`: Directorio de guardado de lotes (`lotes`).
  - `lote_actual`: Ruta del lote activo a auditar (`lotes/lote_actual.csv`).
  - `umbral_drift`: 0.30 (se emite alerta si más del 30% de las variables presentan drift en Evidently).
  - `alfa_evento`: 0.05 (nivel de significancia para prueba chi-cuadrado sobre el target).

---

## Guía de Ejecución de la Actividad 3

### 1. Ejecución de la Suite de Pruebas Unitarias

La suite de pruebas valida la integridad completa de datos, rangos, esquemas, pipelines de modelos, lógica de inferencia, endpoints de la API y algoritmos de monitoreo sin requerir conexión a Docker:

```powershell
poetry run pytest tests/
```

Resultado esperado: 117 tests aprobados (100% pass rate).

---

### 2. Emulación y Comparación de Niveles de Drift (0.0, 0.3 y 0.6)

Para generar la evidencia comparativa requerida por la Actividad 3:

```powershell
poetry run python scripts/evidencia_drift.py
```

Este script ejecuta de manera controlada los tres escenarios y produce:
- **Nivel 0.0 (Sin drift)**: Fracción de variables con drift ~0.0% (Alerta de features: `False`). Proporción de target observada sin diferencia significativa (Alerta de evento: `False`).
- **Nivel 0.3 (Drift moderado)**: Fracción de variables con drift ~23% (por debajo del umbral del 30%, Alerta de features: `False`). Tasa de positivos se desplaza significativamente (Alerta de evento: `True`).
- **Nivel 0.6 (Drift severo)**: Fracción de variables con drift ~51% (supera el umbral del 30%, Alerta de features: `True`). Tasa de positivos fuertemente alterada (p-valor < 0.001, Alerta de evento: `True`).

Los reportes HTML generados por Evidently se almacenan en:
- `reports/drift_features_evidencia_nivel_0.0.html`
- `reports/drift_features_evidencia_nivel_0.3.html`
- `reports/drift_features_evidencia_nivel_0.6.html`
- `reports/evidencia_drift.json`

---

### 3. Ejecución de Pipelines en Prefect

Asegúrate de que los servicios estén activos en Docker (`cd services; docker compose up -d; cd ..`).

#### Pipeline de Entrenamiento
```powershell
poetry run python infraestructure\flows\entrenamiento_flow.py
```
Evalúa los candidatos, registra métricas en MLflow y asigna el alias `champion` al mejor modelo que supere el umbral.

#### Pipeline de Inferencia con Drift Emulado
Configura el nivel deseado en `configs/config.yaml` (`inferencia.nivel_drift: 0.0` o `0.6`) y ejecuta:
```powershell
poetry run python infraestructure\flows\inferencia_flow.py
```
Genera el lote en `lotes/lote_actual.csv`, persiste predicciones en `data/` y registra el run en MLflow.

#### Pipeline de Monitoreo de Features (Evidently)
```powershell
poetry run python -m infraestructure.flows.drift_features_flow
```
O especificando un lote particular:
```powershell
poetry run python -m infraestructure.flows.drift_features_flow --lote lotes/lote_nivel_0.6.csv
```

#### Pipeline de Monitoreo de Evento (Target)
```powershell
poetry run python -m infraestructure.flows.drift_evento_flow
```
O especificando un lote particular:
```powershell
poetry run python -m infraestructure.flows.drift_evento_flow --lote lotes/lote_nivel_0.6.csv
```
*Si se proporciona un lote sin etiquetas, la pipeline rechaza la ejecución indicando con claridad que las etiquetas aún no están disponibles.*

---

### 4. Despliegues Programados en Prefect

Para registrar y servir todos los despliegues de la Actividad 3:

```powershell
poetry run python -m infraestructure.flows.desplegar
```

Despliegues configurados:
- **`entrenamiento-ecommerce`**: Manual (bajo demanda).
- **`inferencia-ecommerce-diaria`**: Diario a las 06:00 UTC (`cron: 0 6 * * *`).
- **`drift-features-ecommerce`**: Diario a las 06:30 UTC (`cron: 30 6 * * *`).
- **`drift-evento-ecommerce`**: Diario a las 07:00 UTC (`cron: 0 7 * * *`).

---

### 5. Servicio de Inferencia en Tiempo Real (FastAPI)

```powershell
poetry run uvicorn infraestructure.api.main:app --port 8000
```

- Documentación interactiva Swagger: [http://localhost:8000/docs](http://localhost:8000/docs)
- Estado y modelo activo: [http://localhost:8000/health](http://localhost:8000/health)

---

## Criterios de Éxito de la Actividad 3

1. **Drift emulado en la inferencia**: Parámetro `nivel_drift` configurable en YAML (`0.0`, `0.3`, `0.6`) que altera variables numéricas, categóricas y la variable objetivo para contrastar escenarios reales de producción.
2. **Detección de Data Drift**: Pipeline automatizada con Evidently que evalúa la distribución multivariada, produce reportes interactivos HTML y activa alertas si la proporción supera `umbral_drift`.
3. **Detección de Target Drift**: Validación estadística de contingencia chi-cuadrado con rechazo explícito y trazabilidad cuando los lotes carecen de etiquetas reales.
4. **Verificación experimental y reportes**: Los niveles `0.0` y `0.6` generan distribuciones y reportes HTML marcadamente diferenciados, reflejando el cambio en el comportamiento del consumidor.
5. **Calidad de software y pruebas**: 117 tests automatizados en `pytest` cubriendo validación de datos, entrenamiento, inferencia, API y monitoreo, integrados en GitHub Actions (`.github/workflows/ci.yml`).
