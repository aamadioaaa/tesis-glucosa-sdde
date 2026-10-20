"""Descriptores no lineales — Capítulo 3 (Secs. 3.4 y 3.6).

* Máximo exponente de Lyapunov por el estimador de Rosenstein
  (``nolds.lyap_r``: emb_dim=3, lag=1, min_tsep=10, trajectory_len=20,
  ajuste polinómico), convertido a h⁻¹ multiplicando por 60/Δt = 4.
* Información mutua promedio (retardo de incrustación) y falsos vecinos más
  cercanos (dimensión de incrustación) para la reconstrucción de Takens.
"""
from __future__ import annotations

import numpy as np

from .datos import DT_MIN


def lyapunov(x: np.ndarray, dt_min: float = DT_MIN) -> float:
    """λ_max [h⁻¹]; ``nan`` si el estimador falla."""
    import nolds
    try:
        lle = nolds.lyap_r(np.asarray(x, dtype=float), emb_dim=3, lag=1, min_tsep=10,
                           trajectory_len=20, fit="poly", debug_plot=False)
        return float(lle * (60.0 / dt_min))
    except Exception:
        return float("nan")


def informacion_mutua(x: np.ndarray, lag_max: int = 24, bins: int = 16) -> np.ndarray:
    """I(τ)/I(0) para τ = 0 … lag_max, con bins adaptativos por cuantiles."""
    x = np.asarray(x, dtype=float)
    bordes = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    out = []
    for lag in range(lag_max + 1):
        a, b = x[:len(x) - lag], x[lag:]
        pab, _, _ = np.histogram2d(a, b, bins=[bordes, bordes])
        pab /= pab.sum()
        pa, pb = pab.sum(1), pab.sum(0)
        nz = pab > 0
        out.append(np.sum(pab[nz] * np.log(pab[nz] / (pa[:, None] * pb[None, :])[nz])))
    out = np.array(out)
    return out / out[0]


def retardo_por_estabilizacion(ami: np.ndarray, tol: float = 0.10) -> int:
    """Primer τ tal que |I(τ) − I(τ−1)| < ``tol``·|I(1) − I(0)| (estabilización de ΔI)."""
    d = np.abs(np.diff(ami))
    for k in range(1, len(d)):
        if d[k] < tol * d[0]:
            return k + 1
    return int(np.argmin(d)) + 1


def falsos_vecinos(x: np.ndarray, tau: int, m_max: int = 8, rtol: float = 15.0,
                   atol: float | None = None) -> np.ndarray:
    """Fracción de falsos vecinos más cercanos (Kennel et al., 1992) para m = 1 … m_max.

    Criterio 1 de Kennel (cociente de distancias > ``rtol``); el criterio 2
    (tamaño del atractor) se agrega sólo si se indica ``atol``.
    """
    from scipy.spatial import cKDTree
    x = np.asarray(x, dtype=float)
    ra = np.std(x)
    out = []
    for m in range(1, m_max + 1):
        n = len(x) - m * tau
        emb = np.column_stack([x[i * tau:i * tau + n] for i in range(m)])
        nxt = x[m * tau:m * tau + n]
        d, idx = cKDTree(emb).query(emb, k=2)
        d, idx = d[:, 1], idx[:, 1]
        ok = d > 0
        dist_extra = np.abs(nxt - nxt[idx])
        f1 = dist_extra[ok] / d[ok] > rtol
        if atol is not None:
            f1 = f1 | (np.sqrt(d[ok] ** 2 + dist_extra[ok] ** 2) / ra > atol)
        out.append(np.mean(f1))
    return np.array(out)


def dimension_por_fnn(fnn: np.ndarray, umbral: float = 0.01) -> int:
    for m, f in enumerate(fnn, start=1):
        if f < umbral:
            return m
    return len(fnn)
