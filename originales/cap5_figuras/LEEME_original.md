# Capítulo 5 — Análisis del error · código de generación de figuras

Contenido de esta carpeta, pensada para vivir en:

```
C:\Users\aru_a\OneDrive\1.Doctorado\9.Tesis\Capitulo_5\Codigo\
```

## Archivos

| Archivo | Qué es |
|---|---|
| `Capitulo_5_Figuras.ipynb` | Notebook documentado, con las figuras ya ejecutadas |
| `capitulo_5_figuras.py` | Mismo código como script (`python capitulo_5_figuras.py`) |
| `datos/` | Acá van las señales por sujeto (ver formato abajo) |
| `datos/_ejemplo_formato.csv` | Ejemplo del formato de columnas esperado |
| `figuras/` | Salida: PNG 300 dpi, PDF vectorial y CSV de las tablas |
| `requisitos.txt` | Dependencias de Python |

## Instalación

Desde una terminal (PowerShell o Anaconda Prompt) parada en esta carpeta:

```powershell
pip install -r requisitos.txt
jupyter lab Capitulo_5_Figuras.ipynb
```

O directamente, sin Jupyter:

```powershell
python capitulo_5_figuras.py
```

## Qué genera

| Figura | Contenido | Archivo de salida |
|---|---|---|
| 5.1 | Análisis global de error — DBT2 (paneles A–I) | `figuras/figura_5_1_DBT2.{png,pdf}` |
| 5.2 | Análisis global de error — NDBT2 (paneles A–I) | `figuras/figura_5_2_NDBT2.{png,pdf}` |
| 5.3 | Comparación DBT2 vs NDBT2 (paneles A–I) | `figuras/figura_5_3_comparativa.{png,pdf}` |

Además exporta `figuras/tabla_5_1_DBT2.csv` y `figuras/tabla_5_2_NDBT2.csv`.

Para insertar en el documento de la tesis conviene el **PDF** (vectorial, no
pixela al ampliar); el PNG a 300 dpi sirve para revisión rápida.

## Los dos modos de datos

En la celda de configuración (o al principio del `.py`):

```python
MODO_DATOS = "sustituto"   # "real" | "sustituto"
```

- **`"sustituto"`** (valor por defecto) — reconstruye señales punto a punto
  compatibles con las métricas publicadas de cada sujeto. Sirve para revisar
  el diseño de las figuras. Los paneles G, H e I son **ilustrativos y no
  publicables** en este modo.
- **`"real"`** — lee las señales de `datos/`.

Los paneles A–F salen siempre de las Tablas 5.1 y 5.2 del capítulo, así que
son idénticos en ambos modos.

## Formato de los datos reales

Un CSV por sujeto en `datos/`:

```
datos/dbt2_01.csv … datos/dbt2_20.csv
datos/ndbt2_01.csv … datos/ndbt2_20.csv
```

Cada uno con 1152 filas (12 días × 96 muestras, Δt = 15 min) y dos columnas:

```
GE,GS
174.2,168.9
...
```

`GE` = glucemia experimental (CGM), `GS` = glucemia simulada por el modelo
SDDE. Si tus archivos tienen otro nombre o estructura, alcanza con editar la
función `cargar_series_reales`.

Después de copiar los datos: poner `MODO_DATOS = "real"` y correr todo de
nuevo (*Kernel → Restart & Run All*). La celda de verificación recalcula las
métricas y las compara contra las filas «Global» de las tablas; si no
coinciden, la discrepancia está en los datos o en la calibración, no en el
graficado.

## Una nota de mantenimiento

Las Tablas 5.1 y 5.2 están transcriptas a mano en el código (listas `_dbt2` y
`_ndbt2`). Si el script que calibró el modelo en el Capítulo 4 ya las produce
como CSV, conviene reemplazar esas listas por una lectura directa de ese
archivo, para no mantener los mismos números en dos lugares.
