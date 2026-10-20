"""Ejecuta el código original del Capítulo 7 y arma las Tablas 7.4 y 7.6 como en la tesis.

Pasos
-----
1. Si no están los CSV originales de CGMacros, los reconstruye a partir de la versión
   compacta del repositorio (`datos/cgmacros/compacto`): por sujeto, `Timestamp`, `Libre GL`
   y `Meal Type`, que son las únicas columnas que leen los scripts.
2. Corre, sin cambios, `extract_St.py` (S(t)) y `consolidar_3grupos.py` (detector de
   ingestas y del fenómeno del alba, plano H×C).
3. Reagrupa por HbA1c de laboratorio (< 5,7 / 5,7–6,4 / ≥ 6,5 %) y excluye 007 y 018, que
   es la versión final del capítulo. El paquete original agrupaba además por glucemia en
   ayunas y usaba los 45 sujetos (archivado en `archivado/agg_3grupos.csv`).

Uso:  python originales/cap7_cgmacros/reproducir.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
COMPACTO = RAIZ / "datos" / "cgmacros" / "compacto"
EXCLUIDOS = (7, 18)          # 007: < 6 días utilizables; 018: > 24 h reconstruidas
GRUPOS = ["Sano", "Prediabético", "Diabetes tipo 2"]


def _grupo(a1c: float) -> str:
    return "Sano" if a1c < 5.7 else ("Prediabético" if a1c < 6.5 else "Diabetes tipo 2")


def preparar_desde_compacto(destino: Path) -> Path:
    """Reconstruye CGMacros-XXX/CGMacros-XXX.csv con las columnas que usan los scripts."""
    destino.mkdir(parents=True, exist_ok=True)
    libre = pd.read_csv(COMPACTO / "libre_1min.csv.gz")
    comidas = pd.read_csv(COMPACTO / "comidas.csv")
    for sid, x in libre.groupby("sujeto"):
        c = comidas[comidas.sujeto == sid]
        d = pd.DataFrame({"Timestamp": x.timestamp, "Libre GL": x.libre_gl, "Meal Type": pd.NA})
        d = pd.concat([d, pd.DataFrame({"Timestamp": c.timestamp, "Libre GL": pd.NA, "Meal Type": c.tipo})])
        d = d.groupby("Timestamp", as_index=False).first().sort_values("Timestamp")
        carpeta = destino / f"CGMacros-{sid:03d}"
        carpeta.mkdir(exist_ok=True)
        d.to_csv(carpeta / f"CGMacros-{sid:03d}.csv", index=False)
    pd.read_csv(COMPACTO / "bio.csv").to_csv(destino / "bio.csv", index=False)
    return destino


def ejecutar(salida: Path | None = None, cgmacros: Path | None = None, verbose: bool = True) -> Path:
    salida = Path(salida or RAIZ / "resultados" / "cap7" / "original")
    if cgmacros is None:
        cgmacros = RAIZ / "datos" / "cgmacros" / "CGMacros"
        if not (cgmacros / "CGMacros-001").exists():
            cgmacros = preparar_desde_compacto(salida / "_cgmacros_desde_compacto")
    env = {**os.environ, "CGMACROS_DIR": str(cgmacros), "SALIDA_CAP7_ORIGINAL": str(salida)}
    for script in ("extract_St.py", "consolidar_3grupos.py", "detect_diabeticos.py", "plot_diab_burgers.py"):
        r = subprocess.run([sys.executable, script], cwd=AQUI, env=env, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr)
        if verbose:
            print(f"$ python {script}\n" + "\n".join(r.stdout.strip().splitlines()[-5:]))
    return salida


def tablas(salida: Path | None = None) -> dict[str, pd.DataFrame]:
    """Tablas 7.4 (detector) y 7.6 (alba) con el agrupamiento y las exclusiones de la tesis."""
    salida = Path(salida or RAIZ / "resultados" / "cap7" / "original")
    R = pd.read_csv(salida / "consolidado_3grupos.csv")
    bio = pd.read_csv(COMPACTO / "bio.csv")
    a1c = dict(zip(bio.subject, bio["A1c PDL (Lab)"]))
    R["sujeto"] = R.subject.str[-3:].astype(int)
    R["grupo_tesis"] = R.sujeto.map(lambda s: _grupo(a1c[s]))
    x = R[~R.sujeto.isin(EXCLUIDOS)]
    g = x.groupby("grupo_tesis")
    t74 = g[["comidas", "detectadas", "detecc", "det_ok"]].sum().loc[GRUPOS]
    t74.columns = ["comidas", "comidas_detectadas", "detecciones", "detecciones_correctas"]
    t74["sensibilidad_%"] = 100 * t74.comidas_detectadas / t74.comidas
    t74["precision_%"] = 100 * t74.detecciones_correctas / t74.detecciones
    t75 = g[["fuerte", "moderada", "leve"]].sum().loc[GRUPOS]
    t76 = g[["alba", "dias"]].sum().loc[GRUPOS]
    t76["eventos_por_dia"] = t76.alba / t76.dias
    return {"consolidado": R, "t74": t74, "t75": t75, "t76": t76}


def modelo_archivado() -> pd.DataFrame:
    """Ajuste archivado del modelo (sim_CGM_diab_resumen.csv), restringido a los 14 DT2 por HbA1c."""
    m = pd.read_csv(AQUI / "archivado" / "sim_CGM_diab_resumen.csv")
    bio = pd.read_csv(COMPACTO / "bio.csv")
    a1c = dict(zip(bio.subject, bio["A1c PDL (Lab)"]))
    return m[m.ID.map(a1c) >= 6.5].reset_index(drop=True)


if __name__ == "__main__":
    s = ejecutar()
    T = tablas(s)
    print(T["t74"].round(1)); print(T["t76"].round(3))
    print(modelo_archivado()[["corr", "rmse", "TIR_real", "TIR_sim"]].mean().round(3))
