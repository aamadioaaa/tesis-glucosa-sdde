"""Rutas de los scripts originales del Capítulo 7 (agregado al publicar el repositorio).

Los scripts originales tenían rutas absolutas del equipo original. Al publicarlos se
reemplazaron por estas dos variables; el resto del código no se modificó.

CGMACROS_DIR           carpeta con CGMacros-XXX/CGMacros-XXX.csv y bio.csv
SALIDA_CAP7_ORIGINAL   carpeta donde se escriben S_t/ y las tablas
"""
import os
from pathlib import Path

_RAIZ = Path(__file__).resolve().parents[2]
_CGM = os.environ.get("CGMACROS_DIR", str(_RAIZ / "datos" / "cgmacros" / "CGMacros"))
_SAL = os.environ.get("SALIDA_CAP7_ORIGINAL", str(_RAIZ / "resultados" / "cap7" / "original"))
os.makedirs(_SAL + "/S_t", exist_ok=True)
