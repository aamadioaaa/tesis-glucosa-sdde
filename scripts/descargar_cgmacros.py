"""Descarga CGMacros (PhysioNet, v1.0.0, CC BY-NC-SA 4.0) y extrae sólo los CSV necesarios.

Uso:  python scripts/descargar_cgmacros.py

Destino: datos/cgmacros/CGMacros/{bio.csv, CGMacros-0XX/CGMacros-0XX.csv}
El ZIP completo (~630 MB, incluye fotografías) se descarga a una carpeta temporal
y se elimina al terminar. No es necesario para ejecutar el notebook del Capítulo 7,
que usa por defecto la versión compacta incluida en datos/cgmacros/compacto/.

Cita: Gutierrez-Osuna, R., Kerr, D., Mortazavi, B., & Das, A. (2025). CGMacros: a
scientific dataset for personalized nutrition and diet monitoring (version 1.0.0).
PhysioNet. https://doi.org/10.13026/3z8q-x658
"""
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

URL = "https://physionet.org/files/cgmacros/1.0.0/CGMacros_dateshifted365.zip"
DESTINO = Path(__file__).resolve().parents[1] / "datos" / "cgmacros" / "CGMacros"


def main():
    DESTINO.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / "cgmacros.zip"
        print(f"Descargando {URL} …")
        with urllib.request.urlopen(URL) as r, open(zpath, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        with zipfile.ZipFile(zpath) as z:
            n = 0
            for info in z.infolist():
                nombre = info.filename
                base = Path(nombre).name
                if base == "bio.csv" or (base.startswith("CGMacros-") and base.endswith(".csv")):
                    destino = DESTINO / base if base == "bio.csv" else DESTINO / base[:-4] / base
                    destino.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(info) as src, open(destino, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    n += 1
    print(f"Listo: {n} archivos en {DESTINO}")


if __name__ == "__main__":
    sys.exit(main())
