"""Ejecuta el script original de las Figuras 5.1–5.3 con las señales reales y simuladas.

El script original (`capitulo_5_figuras.py`) se escribió con dos modos: «sustituto», que
reconstruye señales compatibles con las tablas para revisar el diseño, y «real», que lee un
CSV por serie con las columnas GE (glucosa medida) y GS (glucosa simulada). Este archivo
prepara esos CSV a partir de `datos/` (12 días, 1152 muestras) y corre el script en modo
«real».

Uso:  python originales/cap5_figuras/reproducir.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
sys.path.insert(0, str(RAIZ))


def preparar(destino: Path) -> Path:
    from glucosa import datos
    destino.mkdir(parents=True, exist_ok=True)
    for g in ("DBT2", "NDBT2"):
        for i in range(1, 21):
            ge = datos.cargar_cgm(g, i, datos.N_12_DIAS)
            gs = datos.cargar_simulacion_archivada(g, i)
            pd.DataFrame({"GE": ge, "GS": gs}).to_csv(destino / f"{g.lower()}_{i:02d}.csv", index=False)
    return destino


def ejecutar(salida: Path | None = None) -> Path:
    salida = Path(salida or RAIZ / "resultados" / "cap5" / "original")
    entrada = preparar(salida / "_series")
    env = {**os.environ, "MODO_DATOS": "real", "CAP5_DATOS": str(entrada), "CAP5_FIGURAS": str(salida),
           "MPLBACKEND": "Agg"}
    r = subprocess.run([sys.executable, "capitulo_5_figuras.py"], cwd=AQUI, env=env, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-3000:])
    print("\n".join(r.stdout.strip().splitlines()[-25:]))
    return salida


if __name__ == "__main__":
    ejecutar()
