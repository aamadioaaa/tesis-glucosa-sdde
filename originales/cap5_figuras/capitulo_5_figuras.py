# -*- coding: utf-8 -*-
"""
Capitulo 5 - Analisis del error
Generacion de las Figuras 5.1, 5.2 y 5.3.

Version script del notebook Capitulo_5_Figuras.ipynb.
Uso:  python capitulo_5_figuras.py
Salida: ./figuras/  (PNG 300 dpi + PDF vectorial + CSV de las tablas)
"""


# # Capítulo 5 — Análisis del error: generación de figuras
#
# Notebook reproducible que genera las **Figuras 5.1, 5.2 y 5.3** del Capítulo 5
# (*Análisis del error*) del modelo SDDE de Langevin con retardo.
#
# | Figura | Contenido | Grupo |
# |---|---|---|
# | 5.1 | Análisis global de error (9 paneles A–I) | DBT2 (n = 20) |
# | 5.2 | Análisis global de error (9 paneles A–I) | NDBT2 (n = 20) |
# | 5.3 | Análisis comparativo (9 paneles A–I) | DBT2 vs NDBT2 |
#
# ## Dos modos de ejecución
#
# 1. **Con datos reales** (`MODO_DATOS = "real"`): el notebook lee las señales
#    $G_E(t)$ y $G_S(t)$ de cada sujeto desde `DATA_DIR`. Formato esperado:
#    un CSV por sujeto, `dbt2_01.csv … dbt2_20.csv` y `ndbt2_01.csv … ndbt2_20.csv`,
#    con dos columnas: `GE,GS` (1152 filas, Δt = 15 min).
#    Si tus archivos tienen otro nombre o estructura, solo hay que ajustar la
#    función `cargar_series_reales`.
#
# 2. **Con señales sustitutas** (`MODO_DATOS = "sustituto"`, valor por defecto):
#    como las tablas 5.1 y 5.2 del capítulo están transcriptas íntegramente,
#    el notebook **reconstruye** señales punto a punto compatibles con las
#    métricas publicadas de cada sujeto ($\bar G$, $\sigma_E$, RMSE, $r$, $\beta$).
#    Sirve para verificar el diseño y la estética de las figuras antes de
#    enchufar los datos definitivos.
#
# > ⚠️ **Importante.** Los paneles A–F se calculan siempre a partir de las
# > tablas publicadas del capítulo, de modo que son idénticos en ambos modos.
# > Los paneles G, H e I requieren la serie punto a punto: en modo
# > `"sustituto"` son ilustrativos y **no deben publicarse** como resultados.
# > Cambiá a `MODO_DATOS = "real"` para la versión final de la tesis.


# ── Configuración general ────────────────────────────────────────────────
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats, optimize

import os  # (agregado al publicar: modo y rutas configurables por variables de entorno)
MODO_DATOS = os.environ.get("MODO_DATOS", "sustituto")             # "real" | "sustituto"
DATA_DIR   = Path(os.environ.get("CAP5_DATOS", "./datos"))          # carpeta con los CSV por sujeto (modo "real")
OUT_DIR    = Path(os.environ.get("CAP5_FIGURAS", "./figuras"))      # salida de las figuras
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PUNTOS = 1152                # 12 días × 96 muestras/día
DT_MIN   = 15                  # resolución temporal (min)
N_SUJ    = 20                  # sujetos por grupo
SEED     = 20260722            # semilla para reproducibilidad

DPI      = 300                 # resolución de exportación
FORMATOS = ("png", "pdf")      # formatos de salida

print(f"Modo de datos: {MODO_DATOS}")
print(f"N por grupo  : {N_SUJ} × {N_PUNTOS} = {N_SUJ*N_PUNTOS} puntos")


# ── Estilo gráfico unificado ─────────────────────────────────────────────
mpl.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

COLOR = {
    "DBT2":  "#B03A2E",   # rojo ladrillo
    "NDBT2": "#1F618D",   # azul
    "MAE":   "#2E86C1",
    "RMSE":  "#E67E22",
    "exp":   "#34495E",
    "sim":   "#16A085",
    "ref":   "#7F8C8D",
}

def panel(ax, letra, titulo=""):
    """Rotula un panel con su letra (A, B, C…) y su título."""
    ax.text(-0.13, 1.10, letra, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", ha="left")
    if titulo:
        ax.set_title(titulo, pad=8)
    return ax

def guardar(fig, nombre):
    """Exporta la figura en todos los formatos configurados."""
    for ext in FORMATOS:
        ruta = OUT_DIR / f"{nombre}.{ext}"
        fig.savefig(ruta)
        print(f"  ✓ {os.path.basename(str(ruta))}")


# ## 1. Tablas 5.1 y 5.2 — métricas por sujeto
#
# Transcripción literal de las tablas del capítulo. Columnas:
#
# - `G_media` — glucemia media experimental $\bar G$ (mg/dL)
# - `sigma_E` — desviación estándar experimental (ec. 5.9)
# - `CV` — coeficiente de variación (ec. 5.10)
# - `MAE`, `RMSE` — ecs. 5.3 y 5.4 (mg/dL)
# - `r` — correlación de Pearson (ec. 5.7)
# - `beta` — pendiente de regresión de $G_S$ sobre $G_E$ (ec. 5.8)
# - `P95` — percentil 95 del error absoluto (ec. 5.5)
# - `pct20` — % de puntos con $|e| \le 20$ mg/dL (ec. 5.6)
# - `beta_n` — parámetro estocástico calibrado
# - `tau` — retardo fisiológico (min)
#
# La desviación estándar **simulada** no figura en las tablas, pero se deduce
# exactamente de las ecuaciones 5.7 y 5.8: como
# $\beta = \mathrm{Cov}/\sigma_E^2$ y $r = \mathrm{Cov}/(\sigma_E\sigma_S)$,
# entonces $\sigma_S = \beta\,\sigma_E / r$, y $\Delta\sigma = \sigma_E - \sigma_S$
# (ec. 5.11).


# ── Tabla 5.1 · Grupo DBT2 ───────────────────────────────────────────────
COLS = ["S","G_media","sigma_E","CV","MAE","RMSE","r","beta","P95","pct20","beta_n","tau"]

_dbt2 = [
    ( 1, 174, 55, 31.6, 18, 25, 0.896, 0.785, 52, 68, 0.2940, 36.8),
    ( 2, 122, 40, 33.0, 19, 24, 0.821, 0.789, 46, 59, 0.3119, 86.4),
    ( 3, 120, 25, 21.2, 10, 13, 0.864, 0.694, 26, 91, 0.1013, 19.6),
    ( 4, 117, 21, 17.8, 12, 15, 0.670, 0.443, 31, 87, 0.3537, 33.9),
    ( 5, 161, 52, 32.4, 23, 29, 0.842, 0.610, 55, 52, 0.1900, 67.1),
    ( 6, 117, 21, 17.8, 11, 15, 0.715, 0.505, 28, 88, 0.1604, 37.6),
    ( 7, 111, 25, 22.4, 12, 16, 0.755, 0.575, 32, 82, 0.1932, 47.1),
    ( 8,  98, 25, 25.0, 14, 17, 0.709, 0.566, 33, 77, 0.6667, 48.9),
    ( 9, 103, 23, 22.7, 12, 14, 0.789, 0.615, 27, 84, 0.1149, 58.5),
    (10,  99, 22, 22.3, 11, 14, 0.769, 0.553, 29, 88, 0.1739, 19.6),
    (11, 101, 27, 26.7, 13, 18, 0.765, 0.643, 31, 77, 0.5513, 16.2),
    (12, 123, 22, 18.0, 12, 16, 0.707, 0.650, 31, 82, 0.5756, 41.7),
    (13, 135, 21, 15.6, 10, 12, 0.809, 0.698, 23, 92, 0.3657, 86.7),
    (14, 133, 27, 20.4, 12, 14, 0.858, 0.830, 26, 88, 0.1834, 18.3),
    (15, 113, 22, 19.3, 10, 13, 0.806, 0.607, 26, 89, 0.3237, 31.3),
    (16, 136, 26, 19.2, 13, 16, 0.786, 0.629, 31, 81, 0.3025, 39.9),
    (17, 147, 40, 27.3, 14, 18, 0.894, 0.876, 38, 77, 0.2639, 80.0),
    (18, 116, 23, 20.2, 10, 12, 0.854, 0.701, 23, 92, 0.1006, 22.9),
    (19, 141, 27, 19.2, 12, 16, 0.814, 0.678, 28, 84, 0.1009, 34.9),
    (20, 183, 53, 29.2, 22, 27, 0.867, 0.704, 51, 51, 0.3418, 20.3),
]

# ── Tabla 5.2 · Grupo NDBT2 ──────────────────────────────────────────────
_ndbt2 = [
    ( 1,  85, 12, 14.1,  7, 10, 0.548, 0.335, 21, 95, 0.2499, 69.7),
    ( 2,  77, 14, 18.3,  8, 11, 0.640, 0.548, 24, 92, 0.4592, 20.0),
    ( 3,  90, 15, 17.1,  9, 11, 0.673, 0.521, 22, 93, 0.4757, 18.9),
    ( 4, 106, 17, 16.3, 11, 14, 0.557, 0.306, 28, 89, 0.4497, 15.0),
    ( 5, 100, 13, 13.4,  7, 10, 0.692, 0.526, 19, 95, 0.2384, 87.3),
    ( 6,  84, 10, 11.9,  6,  7, 0.696, 0.530, 14, 99, 0.2954, 17.5),
    ( 7,  83, 17, 20.5, 10, 12, 0.716, 0.653, 25, 91, 0.3912, 15.2),
    ( 8, 100, 18, 18.4, 10, 13, 0.695, 0.500, 27, 90, 0.1011, 16.9),
    ( 9,  68, 21, 31.2, 10, 13, 0.792, 0.723, 26, 86, 0.1293, 69.9),
    (10,  75, 15, 19.7,  9, 11, 0.683, 0.610, 23, 93, 0.4095, 15.2),
    (11, 105, 18, 17.0,  9, 12, 0.733, 0.576, 25, 91, 0.1855, 32.1),
    (12,  91, 22, 24.1, 11, 14, 0.765, 0.647, 30, 85, 0.1587, 16.5),
    (13,  87, 17, 19.6, 10, 13, 0.701, 0.634, 25, 89, 0.3936, 17.0),
    (14,  84, 12, 14.9,  8, 10, 0.634, 0.541, 20, 96, 0.5196, 15.1),
    (15,  92, 14, 15.3,  9, 11, 0.657, 0.512, 20, 96, 0.5484, 19.0),
    (16, 101, 12, 11.7,  7,  9, 0.682, 0.481, 17, 98, 0.2900, 15.0),
    (17,  68, 17, 24.2,  9, 12, 0.694, 0.481, 24, 92, 0.3070, 53.8),
    (18,  93, 12, 13.1,  8, 11, 0.522, 0.353, 22, 94, 0.5053, 23.7),
    (19,  95, 22, 22.6, 12, 15, 0.713, 0.620, 32, 84, 0.1714, 18.1),
    (20,  94, 20, 20.8, 10, 13, 0.750, 0.668, 27, 87, 0.1292, 20.5),
]

def _armar(filas, grupo):
    df = pd.DataFrame(filas, columns=COLS)
    df["grupo"]   = grupo
    df["sigma_S"] = df["beta"] * df["sigma_E"] / df["r"]   # ec. 5.7 y 5.8
    df["dsigma"]  = df["sigma_E"] - df["sigma_S"]          # ec. 5.11
    return df

TAB = {"DBT2": _armar(_dbt2, "DBT2"), "NDBT2": _armar(_ndbt2, "NDBT2")}

# Fila "Global" de cada tabla (N = 20 × 1152 = 23040 puntos)
GLOBAL = {
    "DBT2":  dict(ME=0.0, MAE=13, RMSE=18, r=0.892, beta=0.797, P95=37, pct20=80),
    "NDBT2": dict(ME=0.0, MAE=9,  RMSE=12, r=0.799, beta=0.698, P95=25, pct20=91),
}

TAB["DBT2"].head()


# ## 2. Señales punto a punto ($G_E$, $G_S$)
#
# Los paneles G, H e I necesitan las 1152 observaciones de cada sujeto, no solo
# sus métricas resumen.
#
# **Modo `"real"`.** Se leen de `DATA_DIR`. Ajustá `cargar_series_reales` a tu
# estructura de archivos si hace falta.
#
# **Modo `"sustituto"`.** Se reconstruyen señales consistentes con las tablas:
#
# 1. $G_E(t)$ se genera como una señal tipo CGM (componente circadiana +
#    tres excursiones posprandiales diarias + ruido de Ornstein–Uhlenbeck) y
#    luego se reescala para reproducir **exactamente** $\bar G$ y $\sigma_E$
#    del sujeto.
# 2. El error se construye a partir de la descomposición implícita en las
#    ecuaciones 5.7–5.8. Si $G_S \approx \beta\,G_E + c$ con $c=(1-\beta)\bar G$
#    (sesgo nulo), entonces
#    $\varepsilon$ tiene desviación $\sigma_\varepsilon = \beta\,\sigma_E\sqrt{r^{-2}-1}$
#    y el error resulta $e = (1-\beta)(G_E-\bar G) - \varepsilon$.
# 3. Ese error se deforma con una transformación monótona
#    $e \mapsto \mathrm{sgn}(e)\,|e|^{p}$, con $p$ resuelto numéricamente para
#    reproducir el cociente MAE/RMSE del sujeto (las tablas muestran
#    MAE/RMSE ≈ 0.72–0.75, por debajo del 0.798 gaussiano: colas pesadas,
#    coherente con lo señalado en la sección 5.2 del capítulo).
# 4. Se agrega un sesgo por sujeto, centrado para que el ME global sea nulo,
#    y se reescala el conjunto para reproducir el RMSE tabulado.
# 5. Finalmente $G_S = G_E - e$.


# ── Carga / reconstrucción de las señales ────────────────────────────────

def cargar_series_reales(grupo, s):
    """Lee GE y GS del sujeto `s` (1..20) del grupo indicado.

    Formato esperado: DATA_DIR/{grupo.lower()}_{s:02d}.csv con columnas GE,GS.
    Modificá esta función si tus archivos tienen otra estructura.
    """
    ruta = DATA_DIR / f"{grupo.lower()}_{s:02d}.csv"
    df = pd.read_csv(ruta)
    return df["GE"].to_numpy(float), df["GS"].to_numpy(float)


def _senal_cgm(rng, n=N_PUNTOS, dt_min=DT_MIN):
    """Señal tipo CGM estandarizada (media 0, desvío 1)."""
    t_h = np.arange(n) * dt_min / 60.0
    # Componente circadiana (mínimo nocturno, meseta diurna)
    circ = np.sin(2 * np.pi * (t_h - 9.0) / 24.0)
    # Excursiones posprandiales: subida rápida, decaimiento lento
    comidas = np.zeros(n)
    for dia in range(int(np.ceil(t_h[-1] / 24)) + 1):
        for hora, amp in ((8.0, 1.0), (13.0, 1.25), (20.5, 0.95)):
            t0 = dia * 24 + hora + rng.normal(0, 0.6)
            u = (t_h - t0) / 1.1
            bump = np.where(u > 0, u * np.exp(1.0 - u), 0.0)
            comidas += amp * rng.uniform(0.6, 1.4) * bump
    # Ruido de Ornstein-Uhlenbeck (AR(1) con tiempo de correlación ~90 min)
    phi = np.exp(-dt_min / 90.0)
    ou = np.zeros(n)
    z = rng.normal(size=n)
    for i in range(1, n):
        ou[i] = phi * ou[i - 1] + np.sqrt(1 - phi**2) * z[i]
    x = 1.00 * circ + 1.55 * comidas + 0.80 * ou
    return (x - x.mean()) / x.std()


def _error_sintetico(rng, GE, fila):
    """Error e = GE - GS compatible con beta, r, RMSE y MAE del sujeto."""
    beta, r = fila["beta"], fila["r"]
    sE = GE.std()
    sigma_eps = beta * sE * np.sqrt(1.0 / r**2 - 1.0)
    eps = rng.normal(0.0, sigma_eps, GE.size)
    e = (1.0 - beta) * (GE - GE.mean()) - eps
    e = (e - e.mean()) / e.std()

    # Ajuste de curtosis: e -> sign(e)*|e|^p para igualar el cociente MAE/RMSE
    objetivo = fila["MAE"] / fila["RMSE"]

    def desvio(p):
        y = np.sign(e) * np.abs(e) ** p
        return np.mean(np.abs(y)) / np.sqrt(np.mean(y**2)) - objetivo

    try:
        p = optimize.brentq(desvio, 1.0, 4.0, xtol=1e-4)
    except ValueError:
        p = 1.0
    e = np.sign(e) * np.abs(e) ** p
    e = (e - e.mean()) / np.sqrt(np.mean((e - e.mean()) ** 2))
    return e * fila["RMSE"]


def construir_sustitutos(grupo, seed=SEED):
    """Devuelve {s: (GE, GS)} para los 20 sujetos del grupo."""
    tab = TAB[grupo]
    rng = np.random.default_rng(seed + (0 if grupo == "DBT2" else 1000))
    # Sesgos por sujeto, centrados para que el ME global sea nulo
    sesgos = rng.normal(0, 0.12 * tab["RMSE"].mean(), len(tab))
    sesgos -= sesgos.mean()

    series = {}
    for k, (_, fila) in enumerate(tab.iterrows()):
        GE = fila["G_media"] + fila["sigma_E"] * _senal_cgm(rng)
        e = _error_sintetico(rng, GE, fila) + sesgos[k]
        e *= fila["RMSE"] / np.sqrt(np.mean(e**2))   # RMSE exacto
        series[int(fila["S"])] = (GE, GE - e)
    return series


def obtener_series(grupo):
    if MODO_DATOS == "real":
        return {s: cargar_series_reales(grupo, s) for s in range(1, N_SUJ + 1)}
    return construir_sustitutos(grupo)


SERIES = {g: obtener_series(g) for g in ("DBT2", "NDBT2")}
print({g: (len(v), v[1][0].shape) for g, v in SERIES.items()})


# ── Métricas del capítulo (ecs. 5.1 a 5.11) ──────────────────────────────

def metricas(GE, GS):
    e  = GE - GS                      # ec. 5.1
    ae = np.abs(e)
    return {
        "ME":      e.mean(),                                   # ec. 5.2
        "MAE":     ae.mean(),                                  # ec. 5.3
        "RMSE":    np.sqrt(np.mean(e**2)),                     # ec. 5.4
        "P95":     np.percentile(ae, 95),                      # ec. 5.5
        "pct20":   100 * np.mean(ae <= 20),                    # ec. 5.6
        "pct30":   100 * np.mean(ae <= 30),
        "r":       np.corrcoef(GE, GS)[0, 1],                  # ec. 5.7
        "beta":    np.cov(GE, GS, ddof=0)[0, 1] / np.var(GE),  # ec. 5.8
        "sigma_E": GE.std(),                                   # ec. 5.9
        "sigma_S": GS.std(),
        "G_media": GE.mean(),
        "CV":      100 * GE.std() / GE.mean(),                 # ec. 5.10
        "dsigma":  GE.std() - GS.std(),                        # ec. 5.11
    }


def metricas_por_sujeto(grupo):
    filas = [dict(S=s, **metricas(*SERIES[grupo][s])) for s in sorted(SERIES[grupo])]
    return pd.DataFrame(filas)


def apilar(grupo):
    """Concatena los 20 sujetos: N = 23040 puntos."""
    GE = np.concatenate([SERIES[grupo][s][0] for s in sorted(SERIES[grupo])])
    GS = np.concatenate([SERIES[grupo][s][1] for s in sorted(SERIES[grupo])])
    return GE, GS


MET = {g: metricas_por_sujeto(g) for g in ("DBT2", "NDBT2")}
POOL = {g: apilar(g) for g in ("DBT2", "NDBT2")}

# Verificación: métricas recalculadas vs. valores globales publicados
print("Verificación global (recalculado | publicado)")
for g in ("DBT2", "NDBT2"):
    m, p = metricas(*POOL[g]), GLOBAL[g]
    print(f"  {g:6s} MAE {m['MAE']:5.1f} | {p['MAE']:>4}   "
          f"RMSE {m['RMSE']:5.1f} | {p['RMSE']:>4}   "
          f"r {m['r']:.3f} | {p['r']:.3f}   "
          f"beta {m['beta']:.3f} | {p['beta']:.3f}   "
          f"P95 {m['P95']:5.1f} | {p['P95']:>4}   "
          f"<=20 {m['pct20']:4.0f}% | {p['pct20']:>3}%")


# ## 3. Figuras 5.1 y 5.2 — análisis global por grupo
#
# Nueve paneles según el epígrafe del capítulo:
#
# **(A)** MAE y RMSE por sujeto · **(B)** sesgo ME · **(C)** correlación $r$ ·
# **(D)** % de puntos dentro de ±20 y ±30 mg/dL · **(E)** variabilidad
# experimental y simulada · **(F)** relación MAE–$\sigma_E$ ·
# **(G)** distribución global del error absoluto · **(H)** dispersión global
# experimental–simulada · **(I)** boxplot del error absoluto por sujeto.


# ── Figuras 5.1 y 5.2 ────────────────────────────────────────────────────

def figura_global(grupo, titulo):
    tab  = TAB[grupo]            # tabla publicada (paneles A-F)
    met  = MET[grupo]            # métricas recalculadas (paneles B, D, G-I)
    glob = GLOBAL[grupo]
    GE, GS = POOL[grupo]
    col  = COLOR[grupo]
    S    = tab["S"].to_numpy()
    x    = np.arange(len(S))
    w    = 0.4

    fig, ax = plt.subplots(3, 3, figsize=(16.5, 14.5))
    fig.suptitle(titulo, fontsize=14, fontweight="bold", y=0.985)

    # (A) MAE y RMSE por sujeto
    a = panel(ax[0, 0], "A", "Error absoluto por sujeto")
    a.bar(x - w/2, tab["MAE"],  w, label="MAE",  color=COLOR["MAE"],  edgecolor="white", lw=.5)
    a.bar(x + w/2, tab["RMSE"], w, label="RMSE", color=COLOR["RMSE"], edgecolor="white", lw=.5)
    a.axhline(glob["MAE"],  color=COLOR["MAE"],  ls="--", lw=1.2)
    a.axhline(glob["RMSE"], color=COLOR["RMSE"], ls="--", lw=1.2)
    a.set_ylim(0, 1.32 * tab["RMSE"].max())
    a.set_xticks(x); a.set_xticklabels(S, fontsize=7)
    a.set_xlabel("Sujeto"); a.set_ylabel("mg/dL"); a.legend(ncol=2, loc="upper center")

    # (B) Sesgo ME por sujeto
    b = panel(ax[0, 1], "B", "Sesgo (ME) por sujeto")
    me = met["ME"].to_numpy()
    b.bar(x, me, 0.7, color=np.where(me >= 0, COLOR["MAE"], COLOR["RMSE"]),
          edgecolor="white", lw=.5)
    b.axhline(0, color="k", lw=1)
    lim = max(3.0, 1.25 * np.abs(me).max())
    b.set_ylim(-lim, lim)
    b.set_xticks(x); b.set_xticklabels(S, fontsize=7)
    b.set_xlabel("Sujeto"); b.set_ylabel("ME (mg/dL)")
    b.text(0.98, 0.94, f"ME global = {me.mean():+.1f} mg/dL", transform=b.transAxes,
           ha="right", va="top", fontsize=9,
           bbox=dict(fc="white", ec=COLOR["ref"], alpha=.85, boxstyle="round,pad=0.3"))

    # (C) Correlación de Pearson
    c = panel(ax[0, 2], "C", "Correlación de Pearson por sujeto")
    c.bar(x, tab["r"], 0.7, color=col, alpha=.85, edgecolor="white", lw=.5)
    c.axhline(glob["r"], color="k", ls="--", lw=1.2,
              label=f"r global = {glob['r']:.3f}")
    c.set_ylim(0, 1.18)
    c.set_xticks(x); c.set_xticklabels(S, fontsize=7)
    c.set_xlabel("Sujeto"); c.set_ylabel("r"); c.legend(loc="upper center")

    # (D) Porcentaje dentro de +/- 20 y +/- 30 mg/dL
    d = panel(ax[1, 0], "D", "Puntos dentro de umbrales clínicos")
    d.bar(x - w/2, tab["pct20"],  w, label="|e| ≤ 20 mg/dL", color=col, alpha=.9,
          edgecolor="white", lw=.5)
    d.bar(x + w/2, met["pct30"],  w, label="|e| ≤ 30 mg/dL", color=col, alpha=.45,
          edgecolor="white", lw=.5)
    d.axhline(glob["pct20"], color="k", ls="--", lw=1.2)
    d.set_ylim(0, 128)
    d.set_xticks(x); d.set_xticklabels(S, fontsize=7)
    d.set_xlabel("Sujeto"); d.set_ylabel("% de puntos"); d.legend(ncol=2, loc="upper center")

    # (E) Variabilidad experimental vs simulada
    e = panel(ax[1, 1], "E", "Variabilidad experimental y simulada")
    e.bar(x - w/2, tab["sigma_E"], w, label=r"$\sigma_E$ experimental",
          color=COLOR["exp"], edgecolor="white", lw=.5)
    e.bar(x + w/2, tab["sigma_S"], w, label=r"$\sigma_S$ simulada",
          color=COLOR["sim"], edgecolor="white", lw=.5)
    e.set_ylim(0, 1.42 * max(tab["sigma_E"].max(), tab["sigma_S"].max()))
    e.set_xticks(x); e.set_xticklabels(S, fontsize=7)
    e.set_xlabel("Sujeto"); e.set_ylabel("mg/dL"); e.legend(ncol=2, loc="upper center")
    e.text(0.02, 0.94, r"$\Delta\sigma$ medio = " + f"{tab['dsigma'].mean():.1f} mg/dL",
           transform=e.transAxes, va="top", fontsize=9,
           bbox=dict(fc="white", ec=COLOR["ref"], alpha=.85, boxstyle="round,pad=0.3"))

    # (F) Relación MAE - sigma_E
    f = panel(ax[1, 2], "F", r"Relación MAE – $\sigma_E$")
    sx, sy = tab["sigma_E"].to_numpy(float), tab["MAE"].to_numpy(float)
    f.scatter(sx, sy, s=55, color=col, alpha=.8, edgecolor="white", zorder=3)
    lr = stats.linregress(sx, sy)
    xx = np.linspace(sx.min() * .9, sx.max() * 1.05, 50)
    f.plot(xx, lr.intercept + lr.slope * xx, color="k", ls="--", lw=1.3, zorder=2)
    for xi, yi, si in zip(sx, sy, S):
        f.annotate(str(si), (xi, yi), fontsize=6.5, ha="center", va="center",
                   color="white", zorder=4)
    f.set_xlabel(r"$\sigma_E$ (mg/dL)"); f.set_ylabel("MAE (mg/dL)")
    f.text(0.03, 0.95,
           f"MAE = {lr.slope:.2f}·$\\sigma_E$ + {lr.intercept:.1f}\n"
           f"$R^2$ = {lr.rvalue**2:.3f}   p = {lr.pvalue:.1e}",
           transform=f.transAxes, va="top", fontsize=9,
           bbox=dict(fc="white", ec=COLOR["ref"], alpha=.85, boxstyle="round,pad=0.35"))

    # (G) Distribución global del error absoluto
    g = panel(ax[2, 0], "G", "Distribución global del error absoluto")
    ae = np.abs(GE - GS)
    g.hist(ae, bins=60, color=col, alpha=.75, edgecolor="white", lw=.3)
    g.axvline(glob["MAE"], color=COLOR["MAE"], lw=1.6, ls="--",
              label=f"MAE = {glob['MAE']} mg/dL")
    g.axvline(glob["P95"], color=COLOR["RMSE"], lw=1.6, ls="--",
              label=f"P95 = {glob['P95']} mg/dL")
    g.axvline(20, color="k", lw=1.2, ls=":", label="umbral ±20 mg/dL")
    g.set_xlim(0, np.percentile(ae, 99.7))
    g.set_xlabel("|error| (mg/dL)"); g.set_ylabel("Frecuencia"); g.legend(loc="upper right")

    # (H) Dispersión global experimental - simulada
    h = panel(ax[2, 1], "H", f"Dispersión global (N = {GE.size})")
    hb = h.hexbin(GE, GS, gridsize=55, bins="log", cmap="viridis", mincnt=1, linewidths=0)
    lo = min(GE.min(), GS.min()) - 5
    hi = max(GE.max(), GS.max()) + 5
    h.plot([lo, hi], [lo, hi], color="white", lw=1.6, ls="--", label="identidad")
    lr2 = stats.linregress(GE, GS)
    h.plot([lo, hi], lr2.intercept + lr2.slope * np.array([lo, hi]),
           color="#F1C40F", lw=1.6, label=f"β = {glob['beta']:.3f}")
    h.set_xlim(lo, hi); h.set_ylim(lo, hi); h.set_aspect("equal")
    h.set_xlabel(r"$G_E$ experimental (mg/dL)"); h.set_ylabel(r"$G_S$ simulada (mg/dL)")
    h.legend(loc="upper left", labelcolor="k")
    fig.colorbar(hb, ax=h, fraction=0.046, pad=0.02, label="densidad (log)")
    h.text(0.97, 0.06, f"r = {glob['r']:.3f}", transform=h.transAxes, ha="right",
           fontsize=9, bbox=dict(fc="white", ec=COLOR["ref"], alpha=.85,
                                 boxstyle="round,pad=0.3"))

    # (I) Boxplot del error absoluto por sujeto
    i = panel(ax[2, 2], "I", "Error absoluto por sujeto")
    datos = [np.abs(SERIES[grupo][s][0] - SERIES[grupo][s][1]) for s in sorted(SERIES[grupo])]
    bp = i.boxplot(datos, showfliers=False, patch_artist=True, widths=.62,
                   medianprops=dict(color="k", lw=1.2))
    for caja in bp["boxes"]:
        caja.set(facecolor=col, alpha=.55, edgecolor=col, lw=.8)
    i.axhline(20, color="k", ls=":", lw=1.2, label="umbral ±20 mg/dL")
    i.axhline(glob["MAE"], color=COLOR["MAE"], ls="--", lw=1.2, label="MAE global")
    i.set_ylim(0, 1.35 * max(np.percentile(d_, 75) for d_ in datos))
    i.set_xticks(np.arange(1, len(S) + 1)); i.set_xticklabels(S, fontsize=7)
    i.set_xlabel("Sujeto"); i.set_ylabel("|error| (mg/dL)")
    i.legend(loc="upper center", ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.975])
    return fig


fig51 = figura_global(
    "DBT2",
    "Figura 5.1 · Análisis global de error — Grupo DBT2 "
    "(N = 23040, n = 20, Δt = 15 min)")
guardar(fig51, "figura_5_1_DBT2")
plt.close('all')


fig52 = figura_global(
    "NDBT2",
    "Figura 5.2 · Análisis global de error — Grupo NDBT2 "
    "(N = 23040, n = 20, Δt = 15 min)")
guardar(fig52, "figura_5_2_NDBT2")
plt.close('all')


# ## 4. Figura 5.3 — comparación DBT2 vs NDBT2
#
# **(A)** MAE · **(B)** RMSE · **(C)** % dentro de ±20 mg/dL · **(D)** $\sigma_E$ ·
# **(E)** CV% · **(F)** relación MAE–$\sigma_E$ en ambos grupos ·
# **(G)** boxplot comparativo del error absoluto · **(H)** dispersión global DBT2 ·
# **(I)** dispersión global NDBT2.
#
# Los paneles A–E incluyen el contraste de Mann–Whitney (bilateral, n = 20 por
# grupo), apropiado por el tamaño muestral y la asimetría de las distribuciones
# (cf. Wilcox, 2012).


# ── Figura 5.3 · Comparación entre grupos ────────────────────────────────

def _comparar(ax, letra, col, titulo, ylabel, pct=False):
    a = TAB["DBT2"][col].to_numpy(float)
    b = TAB["NDBT2"][col].to_numpy(float)
    panel(ax, letra, titulo)
    bp = ax.boxplot([a, b], showfliers=False, patch_artist=True, widths=.55,
                    medianprops=dict(color="k", lw=1.4))
    for caja, g in zip(bp["boxes"], ("DBT2", "NDBT2")):
        caja.set(facecolor=COLOR[g], alpha=.35, edgecolor=COLOR[g], lw=1.2)
    rng = np.random.default_rng(7)
    for k, (v, g) in enumerate(zip((a, b), ("DBT2", "NDBT2")), start=1):
        ax.scatter(rng.normal(k, .055, v.size), v, s=26, color=COLOR[g],
                   alpha=.85, edgecolor="white", lw=.5, zorder=3)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["DBT2", "NDBT2"])
    ax.set_ylabel(ylabel)
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    top, bot = max(a.max(), b.max()), min(a.min(), b.min())
    rango = (top - bot) or 1.0
    y = top + 0.10 * rango
    h = 0.03 * rango
    ax.plot([1, 1, 2, 2], [y, y + h, y + h, y], lw=1.1, color="k")
    txt = "p < 0.001" if p < 1e-3 else f"p = {p:.3f}"
    ax.text(1.5, y + 1.4 * h, txt, ha="center", fontsize=9)
    ax.set_ylim(bot - 0.12 * rango, y + 0.20 * rango)
    ax.text(0.02, 0.03, f"medianas: {np.median(a):.1f} vs {np.median(b):.1f}"
            + ("%" if pct else ""),
            transform=ax.transAxes, fontsize=8, color=COLOR["ref"])


def _dispersion(ax, letra, grupo):
    GE, GS = POOL[grupo]
    glob = GLOBAL[grupo]
    panel(ax, letra, f"Dispersión global — {grupo}")
    hb = ax.hexbin(GE, GS, gridsize=52, bins="log", cmap="magma", mincnt=1, linewidths=0)
    lo, hi = min(GE.min(), GS.min()) - 5, max(GE.max(), GS.max()) + 5
    ax.plot([lo, hi], [lo, hi], color="white", lw=1.5, ls="--")
    lr = stats.linregress(GE, GS)
    ax.plot([lo, hi], lr.intercept + lr.slope * np.array([lo, hi]),
            color="#2ECC71", lw=1.6)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel(r"$G_E$ (mg/dL)"); ax.set_ylabel(r"$G_S$ (mg/dL)")
    ax.text(0.03, 0.95,
            f"r = {glob['r']:.3f}\nβ = {glob['beta']:.3f}\n"
            f"MAE = {glob['MAE']} mg/dL",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(fc="white", ec=COLOR["ref"], alpha=.9, boxstyle="round,pad=0.35"))
    ax.figure.colorbar(hb, ax=ax, fraction=0.046, pad=0.02, label="densidad (log)")


def figura_comparativa():
    fig, ax = plt.subplots(3, 3, figsize=(16.5, 14.5))
    fig.suptitle("Figura 5.3 · Análisis comparativo — DBT2 (n = 20) vs NDBT2 (n = 20)",
                 fontsize=14, fontweight="bold", y=0.985)

    _comparar(ax[0, 0], "A", "MAE",     "Error absoluto medio",      "MAE (mg/dL)")
    _comparar(ax[0, 1], "B", "RMSE",    "Raíz del error cuadrático medio", "RMSE (mg/dL)")
    _comparar(ax[0, 2], "C", "pct20",   "Puntos dentro de ±20 mg/dL", "% de puntos", pct=True)
    _comparar(ax[1, 0], "D", "sigma_E", "Variabilidad experimental",  r"$\sigma_E$ (mg/dL)")
    _comparar(ax[1, 1], "E", "CV",      "Coeficiente de variación",   "CV (%)", pct=True)

    # (F) MAE vs sigma_E en ambos grupos
    f = panel(ax[1, 2], "F", r"Relación MAE – $\sigma_E$ por grupo")
    for g in ("DBT2", "NDBT2"):
        sx = TAB[g]["sigma_E"].to_numpy(float)
        sy = TAB[g]["MAE"].to_numpy(float)
        f.scatter(sx, sy, s=52, color=COLOR[g], alpha=.85, edgecolor="white",
                  lw=.6, label=g, zorder=3)
    sx = np.concatenate([TAB[g]["sigma_E"].to_numpy(float) for g in ("DBT2", "NDBT2")])
    sy = np.concatenate([TAB[g]["MAE"].to_numpy(float) for g in ("DBT2", "NDBT2")])
    lr = stats.linregress(sx, sy)
    xx = np.linspace(sx.min() * .9, sx.max() * 1.05, 50)
    f.plot(xx, lr.intercept + lr.slope * xx, color="k", ls="--", lw=1.3, zorder=2)
    f.set_xlabel(r"$\sigma_E$ (mg/dL)"); f.set_ylabel("MAE (mg/dL)")
    f.legend(loc="lower right")
    f.text(0.03, 0.95,
           f"ajuste conjunto: MAE = {lr.slope:.2f}·$\\sigma_E$ + {lr.intercept:.1f}\n"
           f"$R^2$ = {lr.rvalue**2:.3f}",
           transform=f.transAxes, va="top", fontsize=9,
           bbox=dict(fc="white", ec=COLOR["ref"], alpha=.85, boxstyle="round,pad=0.35"))

    # (G) Boxplot comparativo del error absoluto global
    g = panel(ax[2, 0], "G", "Error absoluto global por grupo")
    datos = [np.abs(POOL[k][0] - POOL[k][1]) for k in ("DBT2", "NDBT2")]
    bp = g.boxplot(datos, showfliers=False, patch_artist=True, widths=.5,
                   medianprops=dict(color="k", lw=1.4))
    for caja, k in zip(bp["boxes"], ("DBT2", "NDBT2")):
        caja.set(facecolor=COLOR[k], alpha=.4, edgecolor=COLOR[k], lw=1.2)
    g.axhline(20, color="k", ls=":", lw=1.2, label="umbral ±20 mg/dL")
    for k, (nombre, d) in enumerate(zip(("DBT2", "NDBT2"), datos), start=1):
        g.scatter([k], [GLOBAL[nombre]["MAE"]], marker="D", s=45, zorder=4,
                  color="white", edgecolor="k", lw=1.1)
    g.set_ylim(0, 1.45 * max(np.percentile(d_, 75) for d_ in datos))
    g.set_xticks([1, 2]); g.set_xticklabels(["DBT2", "NDBT2"])
    g.set_ylabel("|error| (mg/dL)")
    g.legend(handles=[Line2D([], [], ls=":", color="k", label="umbral ±20 mg/dL"),
                      Line2D([], [], marker="D", ls="", mfc="white", mec="k",
                             label="MAE global")], loc="upper right")

    # (H) e (I) dispersiones globales
    _dispersion(ax[2, 1], "H", "DBT2")
    _dispersion(ax[2, 2], "I", "NDBT2")

    fig.tight_layout(rect=[0, 0, 1, 0.975])
    return fig


fig53 = figura_comparativa()
guardar(fig53, "figura_5_3_comparativa")
plt.close('all')


# ## 5. Tabla resumen para el texto
#
# Consolida los valores globales de las secciones 5.7, 5.8 y 5.9, incluidos los
# cocientes DBT2/NDBT2 citados en el capítulo (MAE ≈ 1.5×, RMSE ≈ 1.5×).


# ── Tabla resumen y exportación ──────────────────────────────────────────
resumen = pd.DataFrame({
    "DBT2":  GLOBAL["DBT2"],
    "NDBT2": GLOBAL["NDBT2"],
}).T[["ME", "MAE", "RMSE", "P95", "pct20", "r", "beta"]]

cocientes = (resumen.loc["DBT2", ["MAE", "RMSE", "P95"]] /
             resumen.loc["NDBT2", ["MAE", "RMSE", "P95"]]).round(2)

print(resumen.to_string())
print("\nCocientes DBT2 / NDBT2:")
print(cocientes.to_string())

# Exportación de las tablas por sujeto
for g in ("DBT2", "NDBT2"):
    ruta = OUT_DIR / f"tabla_5_{'1' if g == 'DBT2' else '2'}_{g}.csv"
    TAB[g].to_csv(ruta, index=False)
    print(f"  ✓ {os.path.basename(str(ruta))}")

resumen


# ## 6. Cómo pasar a los datos definitivos
#
# 1. Colocá los CSV por sujeto en `./datos/` (`dbt2_01.csv` … `ndbt2_20.csv`),
#    cada uno con columnas `GE,GS` y 1152 filas.
# 2. Cambiá `MODO_DATOS = "real"` en la celda de configuración.
# 3. Ejecutá *Kernel → Restart & Run All*.
# 4. Revisá la celda de verificación: las métricas recalculadas deben coincidir
#    con las filas «Global» de las tablas 5.1 y 5.2. Si no coinciden, la
#    discrepancia está en los datos o en la calibración, no en el graficado.
#
# Las figuras quedan en `./figuras/` en PNG (300 dpi) y PDF vectorial; el PDF es
# el formato recomendado para insertar en el documento final de la tesis.
#
# Si tus tablas fueron generadas por otro script (por ejemplo el que calibró el
# modelo en el Capítulo 4), lo más limpio es reemplazar las listas `_dbt2` y
# `_ndbt2` por una lectura directa del CSV que produjo ese script, para evitar
# mantener los números duplicados en dos lugares.
