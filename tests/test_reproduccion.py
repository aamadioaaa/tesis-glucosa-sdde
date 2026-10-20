"""Pruebas de reproducibilidad: comparan cálculos del paquete con cifras publicadas en la tesis.

Ejecutar desde la raíz del repositorio:  python -m pytest -q
"""
import numpy as np
import pytest

from glucosa import datos, sdde, error, espectral, ordinal, no_lineal
from glucosa.forzamiento import forzamiento

N = datos.N_12_DIAS
SERIES = [(g, i) for g in ("DBT2", "NDBT2") for i in range(1, 21)]


@pytest.mark.parametrize("grupo,i", SERIES)
def test_trayectoria_bit_a_bit(grupo, i):
    """El integrador regenera exactamente la trayectoria archivada (Cap. 4)."""
    g = datos.cargar_cgm(grupo, i, N)
    gs = sdde.simular(g, **datos.parametros(grupo, i))
    assert np.array_equal(np.round(gs), datos.cargar_simulacion_archivada(grupo, i))


def test_tabla_4_2_sujeto_dbt2_01():
    g = datos.cargar_cgm("DBT2", 1, N)
    m = sdde.metricas_ajuste(g, sdde.simular(g, **datos.parametros("DBT2", 1)))
    assert round(m["rho"], 3) == 0.896 and round(m["rmse"], 1) == 24.6 and round(m["sesgo"], 1) == -3.4
    assert round(m["bc_real"], 2) == 2.25 and round(m["bc_sim"], 2) == 2.10 and round(m["dHC"], 3) == 0.058


def test_capitulo_5_global_dbt2():
    E = np.concatenate([datos.cargar_cgm("DBT2", i, N) for i in range(1, 21)])
    S = np.concatenate([datos.cargar_simulacion_archivada("DBT2", i) for i in range(1, 21)])
    m = error.metricas_error(E, S)
    assert round(m["r"], 3) == 0.892 and round(m["b"], 3) == 0.797 and round(m["P95"]) == 37
    z = error.porcentajes_clarke(E, S)
    assert round(z["A"], 1) == 87.2 and round(z["B"], 1) == 12.0 and round(z["D"], 1) == 0.7


def test_capitulo_3_descriptores_ndbt2():
    g = [datos.cargar_cgm("NDBT2", i) for i in datos.IDS_NDBT2_REALES]
    assert round(np.mean([espectral.beta_welch(x) for x in g]), 2) == 1.28
    assert round(np.mean([ordinal.hxc(x)[0] for x in g]), 3) == 0.905
    assert round(ordinal.norma_estacionaria(g[0]), 4) == 0.4820
    assert round(np.mean([no_lineal.lyapunov(x) for x in g]), 3) == 0.274


def test_forzamiento_dimensiones():
    g = datos.cargar_cgm("DBT2", 5, N)
    s, c = forzamiento(g, devolver_componentes=True)
    assert s.shape == g.shape and 0 <= c["E_norm"].min() and c["E_norm"].max() <= 1


def test_duplicado_documentado():
    """DBT2-04 y DBT2-06 son la misma serie (ver REPRODUCIBILIDAD.md)."""
    assert np.array_equal(datos.cargar_cgm("DBT2", 4), datos.cargar_cgm("DBT2", 6))


# --------------------------------------------------------------------------- Capítulo 7 (código original)
def test_cap7_codigo_original_tablas_7_4_y_7_6(tmp_path):
    """Las salidas archivadas del código original, con el agrupamiento final, dan la Tabla 7.4."""
    import shutil
    import sys
    from pathlib import Path
    orig = Path(__file__).resolve().parents[1] / "originales" / "cap7_cgmacros"
    sys.path.insert(0, str(orig))
    import reproducir
    shutil.copy(orig / "archivado" / "consolidado_3grupos.csv", tmp_path)
    T = reproducir.tablas(tmp_path)
    t74 = T["t74"]
    assert t74.comidas.tolist() == [595, 600, 455]
    assert t74["sensibilidad_%"].round(1).tolist() == [49.9, 66.8, 62.6]
    assert t74["precision_%"].round(1).tolist() == [77.5, 76.5, 66.1]
    assert T["t75"].sum().tolist() == [59, 244, 864]
    assert T["t76"].eventos_por_dia.round(3).loc[["Sano", "Diabetes tipo 2"]].tolist() == [0.129, 0.367]
    m = reproducir.modelo_archivado()
    assert len(m) == 14 and round(m["corr"].mean(), 3) == 0.739
