"""Métricas de error del Capítulo 5 y grilla de error de Clarke (Sec. 5.12).

Convención de signo: ε(t) = G_E(t) − G_S(t)   (5.1)
"""
from __future__ import annotations

import numpy as np


def metricas_error(ge: np.ndarray, gs: np.ndarray) -> dict:
    ge = np.asarray(ge, dtype=float)
    gs = np.asarray(gs, dtype=float)
    e = ge - gs
    ae = np.abs(e)
    cov = np.cov(ge, gs, bias=True)[0, 1]
    # Las Tablas 5.1 y 5.2 usan la desviación estándar muestral (ddof = 1) para σ_E y los CV%.
    sE, sS = ge.std(ddof=1), gs.std(ddof=1)
    return dict(
        G_media=ge.mean(), sigma_E=sE, CV_E=100 * sE / ge.mean(),
        CV_S=100 * sS / gs.mean(),
        L1=abs(100 * sE / ge.mean() - 100 * sS / gs.mean()),
        ME=e.mean(), MAE=ae.mean(), RMSE=np.sqrt(np.mean(e ** 2)),
        r=np.corrcoef(ge, gs)[0, 1], b=cov / ge.var(),
        P95=np.quantile(ae, 0.95), P_20=100 * np.mean(ae <= 20), P_30=100 * np.mean(ae <= 30),
        delta_sigma=sE - sS)


def mae_diario(ge, gs, muestras_dia: int = 96) -> np.ndarray:
    e = np.abs(np.asarray(ge, float) - np.asarray(gs, float))
    n = len(e) // muestras_dia
    return e[:n * muestras_dia].reshape(n, muestras_dia).mean(axis=1)


def cv_diario(ge, muestras_dia: int = 96) -> np.ndarray:
    g = np.asarray(ge, float)
    n = len(g) // muestras_dia
    d = g[:n * muestras_dia].reshape(n, muestras_dia)
    return 100 * d.std(axis=1) / d.mean(axis=1)


def zonas_clarke(ref: np.ndarray, est: np.ndarray) -> np.ndarray:
    """Zona A–E de cada par (referencia, estimación) según Clarke et al. (1987)."""
    ref = np.asarray(ref, float)
    est = np.asarray(est, float)
    z = np.empty(len(ref), dtype="<U1")
    for k, (r, y) in enumerate(zip(ref, est)):
        if (r <= 70 and y <= 70) or (0.8 * r <= y <= 1.2 * r):
            z[k] = "A"
        elif (r >= 180 and y <= 70) or (r <= 70 and y >= 180):
            z[k] = "E"
        elif ((70 <= r <= 290) and y >= r + 110) or ((130 <= r <= 180) and (y <= (7 / 5) * r - 182)):
            z[k] = "C"
        elif (r >= 240 and 70 <= y <= 180) or (r <= 175 / 3 and 70 <= y <= 180) or \
             ((175 / 3 <= r <= 70) and y >= (6 / 5) * r):
            z[k] = "D"
        else:
            z[k] = "B"
    return z


def porcentajes_clarke(ref, est) -> dict:
    z = zonas_clarke(ref, est)
    return {k: 100 * np.mean(z == k) for k in "ABCDE"}
