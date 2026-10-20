"""Ejecuta todos los notebooks en orden y regenera resultados/ (≈ 10 min en una PC actual).

Uso:  python scripts/ejecutar_todo.py
"""
import subprocess
import sys
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks"

for nb in sorted(NB.glob("0*.ipynb")):
    print(f"▶ {nb.name}", flush=True)
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                    "--ExecutePreprocessor.timeout=3600", str(nb)], check=True, cwd=NB)
print("✔ Todos los notebooks se ejecutaron sin errores.")
