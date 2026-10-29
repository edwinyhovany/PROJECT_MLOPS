# PROJECT_MLOPS

Repositorio Proyecto MLOPS

## Pasos de ejecución

1. Clonar el repositorio
2. Instalar dependencias con Poetry:
   ```bash
   poetry install
   ```
3. Verificar que el entorno funciona:
   ```bash
   poetry run python -c "import prefect, mlflow, sklearn; print('ok')"
   ```
4. Para agregar nuevas dependencias:
   ```bash
   poetry add <paquete>
   poetry add --group dev <paquete>
   ```

## Estructura

- `pyproject.toml`: Configuración de Poetry y dependencias
- `poetry.lock`: Bloqueo de versiones para reproducibilidad
- `.gitignore`: Carpetas que no se versionan (`mlruns/`, `mlflow.db`, `mlflow_store/`)