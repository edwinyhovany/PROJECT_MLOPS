"""Tests de datos: esquema, nulos, rangos y categorías del dataset E-Commerce."""
import pandas as pd
import pytest

from domain.modelo import OBJETIVO
from domain.validacion import (
    CATEGORIAS_VALIDAS,
    RANGOS,
    validar_datos,
    validar_esquema,
    validar_features,
    validar_nulos,
    validar_rangos,
)
from scripts.load_data import COLUMNAS, cargar_datos


@pytest.fixture(scope="module")
def df():
    return cargar_datos()


@pytest.fixture(scope="module")
def df_crudo():
    return cargar_datos(binarizar=False)


def test_tiene_todas_las_columnas_esperadas(df):
    assert list(df.columns) == COLUMNAS
    assert len(COLUMNAS) == 36


def test_no_hay_nulos(df):
    assert int(df.isna().sum().sum()) == 0


def test_objetivo_es_binario(df):
    assert set(df[OBJETIVO].unique()) <= {0, 1}


def test_objetivo_original_tiene_las_cinco_clases(df_crudo):
    assert set(df_crudo[OBJETIVO].unique()) <= CATEGORIAS_VALIDAS[OBJETIVO]


def test_dataset_tiene_filas_suficientes(df):
    assert len(df) >= 1000


@pytest.mark.parametrize("columna,limites", list(RANGOS.items()))
def test_rangos_de_variables(df, columna, limites):
    minimo, maximo = limites
    if minimo is not None:
        assert df[columna].min() >= minimo
    if maximo is not None:
        assert df[columna].max() <= maximo


@pytest.mark.parametrize("columna,permitidas", list(CATEGORIAS_VALIDAS.items()))
def test_categorias_permitidas(df_crudo, columna, permitidas):
    assert set(df_crudo[columna].unique()) <= permitidas


def test_validar_datos_acepta_el_dataset(df_crudo):
    assert validar_datos(df_crudo) is df_crudo


def test_validar_esquema_rechaza_columna_faltante(df):
    with pytest.raises(ValueError, match="Faltan columnas"):
        validar_esquema(df.drop(columns=["Age"]))


def test_validar_esquema_rechaza_columna_inesperada(df):
    roto = df.copy()
    roto["Columna_Extra"] = 1
    with pytest.raises(ValueError, match="inesperadas"):
        validar_esquema(roto)


def test_validar_nulos_rechaza_nulos(df):
    roto = df.head(20).copy()
    roto.loc[roto.index[0], "Age"] = None
    with pytest.raises(ValueError, match="nulos"):
        validar_nulos(roto)


def test_validar_rangos_rechaza_edad_fuera_de_rango(df):
    roto = df.head(20).copy()
    roto.loc[roto.index[0], "Age"] = 5
    with pytest.raises(ValueError, match="Age"):
        validar_rangos(roto)


def test_validar_rangos_rechaza_categoria_desconocida(df):
    roto = df.head(20).copy()
    roto.loc[roto.index[0], "Region"] = "Marte"
    with pytest.raises(ValueError, match="Region"):
        validar_rangos(roto)


def test_validar_features_sin_objetivo(df):
    validar_features(df.drop(columns=[OBJETIVO]))


def test_validar_features_rechaza_columna_faltante(df):
    with pytest.raises(ValueError, match="Faltan columnas"):
        validar_features(df.drop(columns=[OBJETIVO, "Age"]))


def test_dataset_es_reproducible():
    a = cargar_datos()
    b = cargar_datos()
    pd.testing.assert_frame_equal(a, b)
