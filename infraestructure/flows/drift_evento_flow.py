"""Actividad 3: monitoreo de drift de evento (target).

Compara la proporción de la variable objetivo del lote con la de los datos de entrenamiento
(test chi-cuadrado). Rechaza los lotes sin etiquetas con un mensaje claro, porque en
producción las etiquetas reales pueden llegar con retraso. Emite una alerta si el p-valor
queda por debajo de `monitoreo.alfa_evento`. Guarda un JSON por lote como evidencia.

La lógica está en `domain/monitoreo.py`; este flow solo orquesta.

Ejecución (desde la raíz del proyecto, después de la inferencia):
    poetry run python -m infraestructure.flows.drift_evento_flow
    poetry run python -m infraestructure.flows.drift_evento_flow --lote lotes/lote_nivel_0.6.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Configurar variables de entorno por defecto si no están definidas.
if not os.environ.get("PREFECT_API_URL"):
    os.environ["PREFECT_API_URL"] = "http://localhost:4200/api"

# Raíz del proyecto en sys.path.
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import yaml
from prefect import flow, task

from domain.modelo import OBJETIVO
from domain.monitoreo import (
    dividir,
    drift_de_target,
    guardar_resultado_evento,
    hay_alerta_evento,
    validar_lote,
)
from scripts.load_data import RUTA_CONFIG, cargar_datos, resolver_ruta


@task
def referencia(cfg):
    """Variable objetivo de los datos de entrenamiento."""
    entrenamiento, _ = dividir(cargar_datos(cfg["datos"]["ruta"]), cfg["division"])
    return entrenamiento[OBJETIVO]


@task
def evaluar_evento(referencia_objetivo, ruta_lote, cfg):
    """Compara el target del lote con la referencia. Falla claro si el lote no trae etiquetas."""
    ruta_lote = resolver_ruta(ruta_lote)
    lote = pd.read_csv(ruta_lote)
    validar_lote(lote)
    alfa = cfg["monitoreo"]["alfa_evento"]
    resultado = drift_de_target(referencia_objetivo, lote[OBJETIVO], alfa)

    carpeta = resolver_ruta(cfg["monitoreo"]["carpeta_reportes"])
    ruta_json = carpeta / f"drift_evento_{ruta_lote.stem}.json"
    guardar_resultado_evento(
        resultado,
        ruta_json,
        extra={"lote": str(ruta_lote), "n_filas": int(len(lote)), "alfa": alfa},
    )
    print(
        f"Tasa de {OBJETIVO}: entrenamiento {resultado['tasa_referencia']:.1%}, "
        f"lote {resultado['tasa_lote']:.1%} (p-valor {resultado['p_valor']:.4f})"
    )
    print(f"Resultado guardado en {ruta_json}")
    return resultado


@task
def alerta(resultado, alfa):
    """Avisa si el drift del target es significativo. Devuelve si hay alerta."""
    hay_alerta = hay_alerta_evento(resultado, alfa)
    if hay_alerta:
        print(f"ALERTA: drift del target {OBJETIVO} (p-valor {resultado['p_valor']:.4f} < {alfa})")
    else:
        print(f"Sin alerta: drift del target {OBJETIVO} (p-valor {resultado['p_valor']:.4f} >= {alfa})")
    return hay_alerta


@flow(name="drift-evento-ecommerce", log_prints=True)
def drift_evento_flow(ruta_lote: str | None = None):
    with open(RUTA_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ruta_lote = ruta_lote or cfg["monitoreo"]["lote_actual"]
    resultado = evaluar_evento(referencia(cfg), ruta_lote, cfg)
    return alerta(resultado, cfg["monitoreo"]["alfa_evento"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drift del target de un lote.")
    parser.add_argument("--lote", default=None, help="ruta del lote (por defecto monitoreo.lote_actual)")
    args = parser.parse_args()
    drift_evento_flow(ruta_lote=args.lote)
