"""Calibra la SDDE (Cap. 4) para todas las series de un grupo con Optuna/TPE.

Uso:
    python scripts/calibrar_modelo.py --grupo DBT2 [--ensayos 300] [--con-lle]

Escribe resultados/cap4/recalibracion_<GRUPO>.csv con los parámetros obtenidos y las
métricas de la Tabla 4.2. Las series DBT2 se calibran con el filtro de Savitzky–Golay y
las NDBT2 sin él, como en la calibración archivada. La corrida original no fijó semilla
en el muestreador TPE, por lo que los parámetros resultantes serán cercanos pero no
idénticos a los de datos/modelo/parametros_<GRUPO>.csv.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glucosa import datos, sdde  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grupo", choices=["DBT2", "NDBT2"], required=True)
    ap.add_argument("--ensayos", type=int, default=300)
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--con-lle", action="store_true", help="agrega el término 500·ΔLLE de la ec. (4.23)")
    a = ap.parse_args()
    filtro = a.grupo == "DBT2"
    filas = []
    for i in range(1, 21):
        g = datos.cargar_cgm(a.grupo, i, datos.N_12_DIAS)
        p, _ = sdde.ajustar(g, n_ensayos=a.ensayos, semilla_optuna=a.semilla, filtro_sg=filtro, incluir_lle=a.con_lle)
        m = sdde.metricas_ajuste(g, sdde.simular(g, filtro_sg=filtro, **p))
        filas.append(dict(serie=f"{a.grupo}_{i:02d}", **p, **{k: m[k] for k in ("rho", "rmse", "sesgo", "bc_real", "bc_sim", "dHC", "TIR_real", "TIR_sim")}))
        print(filas[-1]["serie"], {k: round(v, 4) for k, v in p.items()}, f"ρ = {m['rho']:.3f}")
    out = datos.carpeta_resultados(4) / f"recalibracion_{a.grupo}.csv"
    pd.DataFrame(filas).to_csv(out, index=False, float_format="%.10g")
    print("Escrito", out)


if __name__ == "__main__":
    main()
