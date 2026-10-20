"""Indicador de forzamiento metabólico S(t) — Capítulo 3, Secciones 3.3.1–3.3.4.

S(t) = ṽ(t) · [0,2 + 0,8 · √Ê(t)]                                   (3.13)

con
  v(t)  = derivada por diferencias centradas                          (3.6)
  ṽ(t)  = σ_v · tanh( v / (σ_v · γ) ),  γ = 0,5                       (3.7)
  E(t)  = varianza local en ventana deslizante de 10 muestras         (3.8)
  Ê(t)  = envolvente DCT pasa-bajos de E, normalizada por su máximo   (3.9–3.12)

`forzamiento` es la implementación exacta empleada para calibrar el modelo
(Caps. 4–7): reproduce bit a bit los vectores S(t) archivados.

`forzamiento_espectral_cap3` es la variante usada en la Sección 3.3.5 para el
exponente espectral de S(t) (ventana de 17 muestras, corte DCT del 10 %, sin
raíz). Se conserva por separado para reproducir exactamente la Figura 3.7.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.fftpack import dct, idct

from .datos import DT_MIN


def forzamiento(g: np.ndarray, dt: float = DT_MIN, gamma: float = 0.5,
                ventana: int = 10, corte_dct: float = 0.35,
                devolver_componentes: bool = False):
    """S(t) tal como entra en la SDDE (ec. 3.13).

    Parámetros
    ----------
    g : glucosa [mg/dL] muestreada cada ``dt`` minutos.
    gamma : factor de compresión de la tangente hiperbólica.
    ventana : ancho de la ventana de varianza local [muestras].
    corte_dct : fracción de coeficientes DCT conservados (pasa-bajos).
    """
    g = np.asarray(g, dtype=float)
    n = len(g)
    # (3.8) variabilidad local
    e_local = pd.Series(g).rolling(window=ventana, center=True).var().fillna(0).to_numpy()
    # (3.9)–(3.11) suavizado DCT
    c = dct(e_local, type=2, norm="ortho")
    c[int(n * corte_dct):] = 0.0
    e_suave = np.maximum(0.0, idct(c, type=2, norm="ortho"))
    # (3.12) normalización
    e_max = e_suave.max()
    e_norm = e_suave / e_max if e_max > 0 else e_suave
    # (3.6)–(3.7) velocidad acondicionada
    v = np.gradient(g, dt)
    sigma_v = np.std(v) + 1e-6
    v_cond = np.tanh(v / (sigma_v * gamma)) * sigma_v
    # (3.13)
    s = v_cond * (0.2 + 0.8 * np.sqrt(e_norm))
    if devolver_componentes:
        return s, dict(v=v, v_acond=v_cond, E_local=e_local, E_norm=e_norm)
    return s


def forzamiento_espectral_cap3(g: np.ndarray, gamma: float = 0.5,
                               semiancho: int = 8, corte_dct: float = 0.10) -> np.ndarray:
    """Variante de S(t) empleada en la Sección 3.3.5 (exponente espectral de S)."""
    g = np.asarray(g, dtype=float)
    v = np.gradient(g)
    sv = v.std()
    vt = sv * np.tanh(v / (sv * gamma))
    el = np.nan_to_num(pd.Series(g).rolling(2 * semiancho + 1, center=True,
                                           min_periods=1).var().to_numpy())
    cf = dct(el, type=2, norm="ortho")
    cf[max(1, int(corte_dct * len(cf))):] = 0
    rec = idct(cf, type=2, norm="ortho")
    eh = np.clip(rec / rec.max(), 0, 1)
    return vt * (0.2 + 0.8 * eh)
