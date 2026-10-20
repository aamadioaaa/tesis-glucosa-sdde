"""SDDE de Langevin con retardo — Capítulo 4.

    dG/dt = −α (G(t−τ_d) − (B(t) + Δ_b)) + k_f S(t) + β_n η(t)        (4.3)

Integración por método de pasos + Euler–Maruyama (Itô) en una grilla fina de
Δt_sim = 0,5 min, con S(t) mantenido constante en cada intervalo de 15 min,
nivel de referencia B(t) por bloques de 3 días, salvaguardas numéricas
(|F| ≤ 60 mg/dL/min, 40 ≤ G ≤ 600 mg/dL), filtro de Savitzky–Golay de orden 3
(ventana de 15 min = 31 pasos) para emular el sensor y submuestreo a 15 min.

`simular` reproduce bit a bit las 40 trayectorias archivadas en
``datos/modelo/simulaciones_archivadas`` (ruido ``numpy.random.seed(42)``).

Notas de reproducibilidad (ver REPRODUCIBILIDAD.md):
* Las series NDBT2 se calibraron **sin** el filtro de Savitzky–Golay
  (``filtro_sg=False``); las DBT2, con filtro.
* El retardo se discretiza como ⌊τ_d/Δt_sim⌋ pasos (sin interpolación lineal).
* La calibración que produjo los parámetros publicados no incluyó el término
  500·ΔLLE de la ec. (4.23); `ajustar` lo ofrece como opción.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy.signal import savgol_filter

from . import espectral, ordinal
from .datos import DT_MIN, MUESTRAS_DIA
from .forzamiento import forzamiento

DT_SIM = 0.5                    # paso de integración [min]
RAZON = int(DT_MIN / DT_SIM)    # 30 pasos finos por muestra
G_MIN, G_MAX = 40.0, 600.0      # recorte del estado (4.9)
F_MAX = 60.0                    # saturación de la deriva (4.8)
DIAS_BLOQUE = 3
VENTANA_SG = 31                 # 15 min / 0,5 min, impar
SEMILLA_RUIDO = 42

LIMITES = dict(alpha=(0.001, 0.05), tau_d=(15.0, 100.0), k_f=(0.1, 10.0),
               beta_n=(0.1, 1.2), delta_b=(-20.0, 60.0))


@njit(cache=True)
def _integrar(s_fino, alpha, pasos_tau, k_f, beta_n, b_fino, delta_b, g0,
              g_min, g_max, ruido, dt, f_max):
    n = len(s_fino)
    g = np.zeros(n)
    g[:pasos_tau + 1] = g0                       # historia constante φ(t) = G0
    raiz_dt = np.sqrt(dt)
    for t in range(pasos_tau, n - 1):
        f = -alpha * (g[t - pasos_tau] - (b_fino[t] + delta_b)) + k_f * s_fino[t]   # (4.5)
        if f > f_max:
            f = f_max
        if f < -f_max:
            f = -f_max
        g[t + 1] = g[t] + f * dt + beta_n * ruido[t] * raiz_dt                      # (4.7)
        if g[t + 1] < g_min:
            g[t + 1] = g_min
        if g[t + 1] > g_max:
            g[t + 1] = g_max
    return g


def referencia_por_bloques(g: np.ndarray, dias_bloque: int = DIAS_BLOQUE) -> np.ndarray:
    """B(t): media de la serie observada en bloques de ``dias_bloque`` días (grilla de 15 min)."""
    g = np.asarray(g, dtype=float)
    b = np.zeros(len(g))
    tam = MUESTRAS_DIA * dias_bloque
    for i0 in range(0, len(g), tam):
        b[i0:i0 + tam] = np.mean(g[i0:i0 + tam])
    return b


def ruido_blanco(n_muestras: int, semilla: int = SEMILLA_RUIDO) -> np.ndarray:
    np.random.seed(semilla)
    return np.random.randn(n_muestras * RAZON).astype(np.float64)


def simular(g_obs: np.ndarray, alpha: float, tau_d: float, k_f: float, beta_n: float,
            delta_b: float, filtro_sg: bool = True, s: np.ndarray | None = None,
            b: np.ndarray | None = None, ruido: np.ndarray | None = None,
            g0: float | None = None, g_min: float = G_MIN, devolver_fina: bool = False,
            **_):
    """Trayectoria simulada a 15 min.

    Por defecto S(t) y B(t) se calculan a partir de ``g_obs``; pueden pasarse
    explícitamente (Capítulo 6: forzamiento intervenido y referencia causal).
    """
    g_obs = np.asarray(g_obs, dtype=float)
    n = len(g_obs)
    s = forzamiento(g_obs) if s is None else np.asarray(s, dtype=float)
    b = referencia_por_bloques(g_obs) if b is None else np.asarray(b, dtype=float)
    ruido = ruido_blanco(n) if ruido is None else ruido
    g0 = g_obs[0] if g0 is None else g0
    s_f = np.repeat(s, RAZON)[:n * RAZON].astype(np.float64)
    b_f = np.repeat(b, RAZON)[:n * RAZON].astype(np.float64)
    fina = _integrar(s_f, alpha, int(tau_d / DT_SIM), k_f, beta_n, b_f, delta_b, g0,
                     g_min, G_MAX, ruido, DT_SIM, F_MAX)
    if filtro_sg:
        suave = savgol_filter(fina, window_length=VENTANA_SG, polyorder=3, mode="nearest")
        suave[0] = fina[0]
    else:
        suave = fina
    out = suave[::RAZON][:n]
    return (out, fina) if devolver_fina else out


# --------------------------------------------------------------------------- métricas
def metricas_clinicas(g: np.ndarray) -> dict:
    g = np.asarray(g, dtype=float)
    n = len(g)
    return dict(TBR=np.sum(g < 70) / n, TIR=np.sum((g >= 70) & (g <= 180)) / n,
                TAR=np.sum(g > 180) / n, TBR54=np.sum(g < 54) / n)


def metricas_ajuste(g_real: np.ndarray, g_sim: np.ndarray) -> dict:
    """Métricas de la Tabla 4.2 (calculadas sobre la trayectoria sin redondear)."""
    hr, cr = ordinal.hxc(g_real)
    hs, cs = ordinal.hxc(g_sim)
    cr_, cs_ = metricas_clinicas(g_real), metricas_clinicas(g_sim)
    return dict(
        rho=float(np.corrcoef(g_sim, g_real)[0, 1]),
        rmse=float(np.sqrt(np.mean((g_sim - g_real) ** 2))),
        sesgo=float(np.mean(g_sim) - np.mean(g_real)),
        bc_real=espectral.beta_periodograma(g_real), bc_sim=espectral.beta_periodograma(g_sim),
        dHC=float(np.hypot(hs - hr, cs - cr)),
        TIR_real=100 * cr_["TIR"], TIR_sim=100 * cs_["TIR"],
        H_real=hr, C_real=cr, H_sim=hs, C_sim=cs)


# --------------------------------------------------------------------------- calibración
def costo(g_real, g_sim, objetivos, incluir_lle=False):
    """Función objetivo J(θ) (4.23) tal como se evaluó en la calibración."""
    from .no_lineal import lyapunov
    if np.any(g_sim < 40.0):
        return 5000.0 + 100.0 * np.sum(40.0 - g_sim[g_sim < 40.0])
    if np.std(g_sim) < 1.0:
        return 4000.0
    c = metricas_clinicas(g_sim)
    rmse = np.sqrt(np.mean((g_sim - g_real) ** 2))
    rho = np.corrcoef(g_sim, g_real)[0, 1]
    rho = 0.0 if np.isnan(rho) else rho
    std_err = abs(np.std(g_sim) - np.std(g_real)) / (np.std(g_real) + 1e-6)
    h, cc = ordinal.hxc(g_sim)
    j = (100 * rmse + 150 * (1 - rho) + 100 * std_err
         + 1000 * abs(objetivos["TBR"] - c["TBR"]) + 400 * abs(objetivos["TIR"] - c["TIR"])
         + 400 * abs(objetivos["TAR"] - c["TAR"])
         + 250 * abs(objetivos["bc"] - espectral.beta_periodograma(g_sim))
         + 600 * np.hypot(h - objetivos["H"], cc - objetivos["C"])
         + 30 * abs(np.mean(g_sim) - np.mean(g_real)))
    if incluir_lle:
        j += 500 * abs(objetivos["LLE"] - lyapunov(g_sim))
    return float(j)


def ajustar(g_real: np.ndarray, n_ensayos: int = 300, semilla_optuna: int | None = 42,
            filtro_sg: bool = True, incluir_lle: bool = False, mostrar_progreso: bool = False):
    """Calibra θ = (α, τ_d, k_f, β_n, Δ_b) con Optuna/TPE.

    La calibración original se ejecutó con ``optuna.create_study()`` sin semilla
    del muestreador; por eso una nueva corrida converge a parámetros cercanos
    pero no idénticos. Para reproducir exactamente las cifras de la tesis se
    usan los parámetros archivados (``datos.parametros``).
    """
    import optuna
    from .no_lineal import lyapunov
    g_real = np.asarray(g_real, dtype=float)
    s = forzamiento(g_real)
    b = referencia_por_bloques(g_real)
    ruido = ruido_blanco(len(g_real))
    h, c = ordinal.hxc(g_real)
    cl = metricas_clinicas(g_real)
    obj = dict(TBR=cl["TBR"], TIR=cl["TIR"], TAR=cl["TAR"], H=h, C=c,
               bc=espectral.beta_periodograma(g_real),
               LLE=lyapunov(g_real) if incluir_lle else np.nan)

    def objetivo(trial):
        p = {k: trial.suggest_float(k, *LIMITES[k]) for k in LIMITES}
        try:
            g_sim = simular(g_real, s=s, b=b, ruido=ruido, filtro_sg=filtro_sg, **p)
            return costo(g_real, g_sim, obj, incluir_lle)
        except Exception:
            return 8000.0

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=semilla_optuna)
    estudio = optuna.create_study(direction="minimize", sampler=sampler)
    estudio.optimize(objetivo, n_trials=n_ensayos, show_progress_bar=mostrar_progreso)
    return estudio.best_params, estudio
