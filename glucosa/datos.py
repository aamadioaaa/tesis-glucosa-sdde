"""Carga de datos del repositorio.

Convenciones
------------
* Registros propios: 1200 muestras cada 15 min (12,5 días). Los Capítulos 4–6
  usan las primeras 1152 (12 días completos).
* DBT2_01 … DBT2_20  : registros reales con diabetes tipo 2.
* NDBT2_01 … NDBT2_05: registros reales sin diabetes tipo 2.
* NDBT2_06 … NDBT2_20: realizaciones sintéticas (aumento DCT + PCA + GMM).
  Se excluyen de todo contraste inferencial.

Equivalencia con los nombres de archivo usados durante la tesis:
``subject_i_E.csv`` → ``DBT2_ii`` y ``subject_i_S.csv`` → ``NDBT2_ii``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "datos"
RESULTADOS = RAIZ / "resultados"

DT_MIN = 15.0          # intervalo de muestreo [min]
MUESTRAS_DIA = 96
N_12_DIAS = 12 * MUESTRAS_DIA   # 1152

IDS_DBT2 = list(range(1, 21))
IDS_NDBT2_REALES = [1, 2, 3, 4, 5]
IDS_NDBT2_AUMENTADOS = list(range(6, 21))

#: DBT2_04 y DBT2_06 contienen la misma serie (ver REPRODUCIBILIDAD.md).
DUPLICADOS_DBT2 = {6: 4}


def _carpeta(grupo: str, i: int) -> Path:
    grupo = grupo.upper()
    if grupo == "DBT2":
        return DATOS / "cgm" / "DBT2"
    if grupo == "NDBT2":
        sub = "NDBT2" if i in IDS_NDBT2_REALES else "NDBT2_aumentado"
        return DATOS / "cgm" / sub
    raise ValueError("grupo debe ser 'DBT2' o 'NDBT2'")


def cargar_cgm(grupo: str, i: int, n: int | None = None) -> np.ndarray:
    """Serie de glucosa [mg/dL] del registro ``grupo_i``.

    ``n=None`` devuelve las 1200 muestras; ``n=1152`` los 12 días completos.
    """
    ruta = _carpeta(grupo, i) / f"{grupo.upper()}_{i:02d}.csv"
    g = pd.read_csv(ruta)["glucosa_mg_dl"].to_numpy(dtype=float)
    return g if n is None else g[:n]


def cargar_parametros(grupo: str) -> pd.DataFrame:
    """Parámetros calibrados de la SDDE (Cap. 4), con precisión completa."""
    ruta = DATOS / "modelo" / f"parametros_{grupo.upper()}.csv"
    return pd.read_csv(ruta, float_precision="round_trip")


def parametros(grupo: str, i: int) -> dict:
    fila = cargar_parametros(grupo).iloc[i - 1]
    return dict(alpha=float(fila.alpha), tau_d=float(fila.tau_d_min), k_f=float(fila.k_f),
                beta_n=float(fila.beta_n), delta_b=float(fila.delta_b),
                filtro_sg=bool(fila.filtro_sg))


def cargar_simulacion_archivada(grupo: str, i: int) -> np.ndarray:
    """Trayectoria simulada tal como quedó archivada al calibrar (enteros)."""
    ruta = DATOS / "modelo" / "simulaciones_archivadas" / f"{grupo.upper()}_{i:02d}_sim.csv"
    return pd.read_csv(ruta)["glucosa_simulada_mg_dl"].to_numpy(dtype=float)


def carpeta_resultados(capitulo: int) -> Path:
    p = RESULTADOS / f"cap{capitulo}"
    p.mkdir(parents=True, exist_ok=True)
    return p
