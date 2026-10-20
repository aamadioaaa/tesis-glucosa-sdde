"""Análisis espectral y de escala — Capítulo 3 (Secs. 3.2.2–3.2.4) y Capítulo 4 (ec. 4.20).

* `beta_welch`: exponente βc por Welch (256 puntos, 50 % de solapamiento) en la
  banda confiable 1–12 h (0,08–1 ciclos/h).
* `beta_periodograma`: exponente usado dentro de la función objetivo J(θ)
  del Capítulo 4 (periodograma FFT completo, ajuste log-log).
* `dfa`: análisis de fluctuaciones sin tendencia de orden 1, escalas 8–288 muestras.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter, welch

FS = 4.0            # muestras por hora (Δt = 15 min)
NPERSEG = 256
BANDA_CONFIABLE = (0.08, 1.0)   # ciclos/h ↔ períodos de 12 h a 1 h


def psd(x: np.ndarray, fs: float = FS, nperseg: int = NPERSEG):
    x = np.asarray(x, dtype=float)
    return welch(x - x.mean(), fs=fs, nperseg=nperseg, detrend=False)


def beta_welch(x: np.ndarray, banda=BANDA_CONFIABLE) -> float:
    """βc = −pendiente del ajuste log-log de la PSD de Welch en ``banda``."""
    f, p = psd(x)
    m = (f > 0) & (f >= banda[0]) & (f <= banda[1])
    return float(-np.polyfit(np.log(f[m]), np.log(p[m]), 1)[0])


def beta_banda_completa(x: np.ndarray) -> float:
    return beta_welch(x, banda=(1e-9, FS / 2))


def beta_periodograma(x: np.ndarray, dt: float = 15.0) -> float:
    """Exponente espectral tal como se calcula en J(θ) (Cap. 4, ec. 4.20–4.21)."""
    x = np.asarray(x, dtype=float)
    f = np.fft.rfftfreq(len(x), d=dt)
    p = np.abs(np.fft.rfft(x)) ** 2
    keep = (f > 0) & (f < 0.5 / dt)
    pendiente, _ = np.polyfit(np.log10(f[keep]), np.log10(p[keep]), 1)
    return float(-pendiente)


ESCALAS_DFA = np.unique(np.round(np.geomspace(8, 288, 16)).astype(int))


def dfa(x: np.ndarray, orden: int = 1, escalas=ESCALAS_DFA) -> float:
    """Exponente α de DFA (orden 1, escalas ≈ 2–72 h)."""
    x = np.asarray(x, dtype=float)
    y = np.cumsum(x - x.mean())
    F, us = [], []
    for n in escalas:
        nw = len(y) // n
        if nw < 2:
            continue
        t = np.arange(n)
        r = [np.mean((y[i * n:(i + 1) * n]
                      - np.polyval(np.polyfit(t, y[i * n:(i + 1) * n], orden), t)) ** 2)
             for i in range(nw)]
        F.append(np.sqrt(np.mean(r)))
        us.append(n)
    return float(np.polyfit(np.log(us), np.log(F), 1)[0])


def residuo_fluctuante(x: np.ndarray, ventana: int = 49, orden: int = 3) -> np.ndarray:
    """Componente fluctuante: x menos su tendencia lenta (Savitzky–Golay)."""
    x = np.asarray(x, dtype=float)
    return x - savgol_filter(x, ventana, orden)
