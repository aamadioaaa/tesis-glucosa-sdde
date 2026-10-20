"""Genera la versión compacta de CGMacros incluida en el repositorio.

Lee los CSV originales (scripts/descargar_cgmacros.py) y escribe en
datos/cgmacros/compacto/:
  libre_1min.csv.gz  sujeto, timestamp, libre_gl  (filas publicadas del sensor Libre)
  comidas.csv        sujeto, timestamp, tipo      (marcas temporales de ingesta)
  bio.csv            subject, A1c PDL (Lab), Fasting GLU - PDL (Lab)

Es una adaptación de CGMacros (CC BY-NC-SA 4.0) y se distribuye bajo la misma licencia.
"""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
ORIG = RAIZ / "datos" / "cgmacros" / "CGMacros"
DEST = RAIZ / "datos" / "cgmacros" / "compacto"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    libre, comidas = [], []
    for f in sorted(ORIG.glob("CGMacros-0*/CGMacros-0*.csv")):
        sid = int(f.stem.split("-")[1])
        d = pd.read_csv(f, low_memory=False)
        d.columns = [c.strip() for c in d.columns]
        x = d[["Timestamp", "Libre GL"]].dropna()
        libre.append(pd.DataFrame({"sujeto": sid, "timestamp": x["Timestamp"], "libre_gl": x["Libre GL"]}))
        m = d[d["Meal Type"].notna() & (d["Meal Type"].astype(str).str.strip() != "")]
        comidas.append(pd.DataFrame({"sujeto": sid, "timestamp": m["Timestamp"], "tipo": m["Meal Type"].astype(str).str.strip()}))
    pd.concat(libre).to_csv(DEST / "libre_1min.csv.gz", index=False, compression="gzip")
    pd.concat(comidas).to_csv(DEST / "comidas.csv", index=False)
    bio = pd.read_csv(ORIG / "bio.csv")
    bio.columns = [c.strip() for c in bio.columns]
    bio[["subject", "A1c PDL (Lab)", "Fasting GLU - PDL (Lab)"]].to_csv(DEST / "bio.csv", index=False)
    print("Compacto escrito en", DEST)


if __name__ == "__main__":
    main()
