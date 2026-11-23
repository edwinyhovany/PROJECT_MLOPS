"""Evidencia reproducible de drift para los niveles 0.0, 0.3 y 0.6.

Entrena un modelo local (sin MLflow), puntúa un lote de prueba por nivel, guarda las copias
en `lotes/`, calcula el reporte de features con Evidently y el drift del target, y escribe
`reports/evidencia_drift.json`.

No sustituye a la pipeline de inferencia: sirve para comparar los tres niveles sin levantar
servicios y dejar evidencia en la memoria y en el documento .docx.

Ejecución (desde la raíz del proyecto):
    poetry run python scripts/evidencia_drift.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from domain.entrenamiento import dividir_datos, entrenar_y_evaluar
from domain.inferencia import guardar_lote, preparar_lote, puntuar
from domain.modelo import OBJETIVO, VARIABLES
from domain.monitoreo import (
    calcular_drift_features,
    drift_de_target,
    hay_alerta_evento,
    hay_alerta_features,
)
from scripts.load_data import cargar_config, cargar_datos, resolver_ruta

NIVELES = [0.0, 0.3, 0.6]


def main() -> None:
    cfg = cargar_config()
    df = cargar_datos(cfg["datos"]["ruta"])
    division = cfg["division"]
    train, _ = dividir_datos(df, division)

    # Modelo local para puntuar; el champion real lo aporta MLflow en la pipeline.
    features = next(f for f in cfg["features"] if f["nombre"] == "all")
    resultado = entrenar_y_evaluar(
        df,
        division,
        features,
        "logistic_regression",
        {"C": 1.0, "max_iter": 1000},
        cfg["metricas"]["umbral"],
        nombre_run="evidencia-drift",
    )
    print(f"Modelo local: AUC={resultado.auc:.4f}")

    referencia_features = train[VARIABLES]
    referencia_target = train[OBJETIVO]
    carpeta_reportes = resolver_ruta(cfg["monitoreo"]["carpeta_reportes"])
    carpeta_lotes = resolver_ruta(cfg["monitoreo"]["lotes_dir"])
    carpeta_predicciones = resolver_ruta(cfg["inferencia"]["carpeta_salida"])
    carpeta_predicciones.mkdir(parents=True, exist_ok=True)
    carpeta_reportes.mkdir(parents=True, exist_ok=True)

    resumen = {
        "modelo": {"tipo": "logistic_regression", "auc": resultado.auc},
        "division": division,
        "tamano_lote": cfg["inferencia"]["tamano_lote"],
        "umbral_drift": cfg["monitoreo"]["umbral_drift"],
        "alfa_evento": cfg["monitoreo"]["alfa_evento"],
        "niveles": [],
    }

    base_fecha = datetime.now()
    for i, nivel in enumerate(NIVELES):
        lote = preparar_lote(
            df,
            division,
            cfg["inferencia"]["tamano_lote"],
            nivel_drift=nivel,
            semilla=division["random_state"],
        )
        puntuado = puntuar(resultado.modelo, lote, cfg["metricas"]["umbral"])
        rutas = guardar_lote(lote, carpeta_lotes, nivel, fecha=base_fecha + timedelta(minutes=i))

        ruta_html = carpeta_reportes / f"drift_features_{Path(rutas['historico']).stem}.html"
        fraccion, _ = calcular_drift_features(referencia_features, lote, ruta_html)
        evento = drift_de_target(referencia_target, lote[OBJETIVO], cfg["monitoreo"]["alfa_evento"])
        ruta_evento = carpeta_reportes / f"drift_evento_{Path(rutas['historico']).stem}.json"
        ruta_evento.write_text(
            json.dumps({"lote": rutas["historico"], "n_filas": len(lote), **evento}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        ruta_prediccion = carpeta_predicciones / f"prediccion_evidencia_nivel_{nivel:.1f}.csv"
        puntuado.to_csv(ruta_prediccion, index=False)

        alerta_features = hay_alerta_features(fraccion, cfg["monitoreo"]["umbral_drift"])
        alerta_evento = hay_alerta_evento(evento, cfg["monitoreo"]["alfa_evento"])
        item = {
            "nivel": nivel,
            "lote": rutas["historico"],
            "prediccion": str(ruta_prediccion),
            "n_filas": len(lote),
            "tasa_target_lote": evento["tasa_lote"],
            "fraccion_features_con_drift": fraccion,
            "alerta_features": alerta_features,
            "reporte_features": str(ruta_html),
            "p_valor_target": evento["p_valor"],
            "alerta_target": alerta_evento,
            "reporte_evento": str(ruta_evento),
        }
        resumen["niveles"].append(item)
        print(
            f"Nivel {nivel:.1f}: drift features {fraccion:.0%} "
            f"(alerta={alerta_features}), tasa target {evento['tasa_lote']:.1%} "
            f"(p={evento['p_valor']:.4g}, alerta={alerta_evento})"
        )

    ruta_resumen = carpeta_reportes / "evidencia_drift.json"
    ruta_resumen.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Resumen de evidencia: {ruta_resumen}")


if __name__ == "__main__":
    main()
