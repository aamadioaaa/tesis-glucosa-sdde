"""Análisis ordinal de Bandt–Pompe — Capítulo 3 (Secs. 3.5) y Capítulo 6 (Sec. 6.8).

* Patrones ordinales D = 3, retardo 1 (orden estable: ``argsort`` mergesort).
* Entropía de permutación normalizada H y complejidad estadística de
  Jensen–Shannon C (Lamberti et al., 2004).
* OPTM (matriz de transición de patrones ordinales) con las tres reglas de
  empates de Amadio et al. (2023, Physica A): omisión, secuencial y ponderada;
  norma de Frobenius, norma del vector estacionario ‖v‖ e índice compuesto ν.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import eig

PERM = {(0, 1, 2): 0, (0, 2, 1): 1, (1, 0, 2): 2, (1, 2, 0): 3, (2, 0, 1): 4, (2, 1, 0): 5}
NOMBRES_PATRONES = ["123", "132", "213", "231", "312", "321"]

# Centroides y umbrales del clasificador de Amadio et al. (2023)
CENTROIDE_N = np.array([0.9060, 0.0816])
CENTROIDE_P = np.array([0.7901, 0.1619])
ETA = 0.06
UMBRALES = {
    "omision":    {"fN": 1.84, "fP": 1.85, "sN": 0.49, "sP": 0.50},
    "secuencial": {"fN": 1.87, "fP": 1.89, "sN": 0.49, "sP": 0.50},
    "ponderado":  {"fN": 1.87, "fP": 1.89, "sN": 0.52, "sP": 0.53},
}


# --------------------------------------------------------------------------- H × C
def complejidad_estadistica(p: np.ndarray) -> tuple[float, float]:
    """(H, C) para una distribución de probabilidad ``p`` de N estados."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    pe = np.ones(n) / n
    nz = p[p > 0]
    if len(nz) == 0:
        return 0.0, 0.0

    def ent(q):
        q = q[q > 0]
        return -np.sum(q * np.log(q))

    h = ent(p) / np.log(n)
    jsd = ent(0.5 * (p + pe)) - 0.5 * ent(p) - 0.5 * ent(pe)
    k = (n + 1) / n
    q0 = -2.0 / (k * np.log(n + 1) - 2 * np.log(2 * n) + np.log(n))
    return float(h), float(h * jsd * q0)


def indice_patron(seg) -> int:
    return PERM[tuple(np.argsort(seg, kind="mergesort").tolist())]


def distribucion_ordinal(x: np.ndarray, d: int = 3, tau: int = 1) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    cuenta = np.zeros(math.factorial(d))
    for t in range(len(x) - (d - 1) * tau):
        cuenta[indice_patron(x[t:t + d * tau:tau])] += 1
    return cuenta / cuenta.sum()


def hxc(x: np.ndarray) -> tuple[float, float]:
    """(H, C) de una serie, D = 3, τ = 1."""
    return complejidad_estadistica(distribucion_ordinal(x))


def curvas_limite(m: int = 3):
    """Curvas de complejidad mínima y máxima del plano H × C."""
    n = math.factorial(m)
    hmin, cmin, hmax, cmax = [], [], [], []
    for p in np.linspace(0.001, 0.999, 200):
        P = np.array([p] + list((1 - p) / (n - 1) * np.ones(n - 1)))
        h, c = complejidad_estadistica(P)
        hmax.append(h); cmax.append(c)
    for k in range(1, n):
        for p in np.linspace(1 / k, 1 / (k + 1), 50):
            P = np.zeros(n); P[:k] = p
            if 1 - k * p > 0:
                P[k] = 1 - k * p
            h, c = complejidad_estadistica(P)
            hmin.append(h); cmin.append(c)
    return np.array(hmin), np.array(cmin), np.array(hmax), np.array(cmax)


# --------------------------------------------------------------------------- OPTM
def _pesos_ponderados(seg) -> np.ndarray:
    """Vector de pesos de la regla ponderada (Tabla 2 de Amadio et al., 2023)."""
    w = np.zeros(6)
    s0, s1, s2 = seg
    tol = 1e-8
    if abs(s0 - s1) < tol and abs(s1 - s2) < tol:
        w[:] = 1 / 6
        return w
    if abs(s0 - s1) < tol:
        if s0 < s2: w[0] = w[2] = 0.5
        else:       w[4] = w[5] = 0.5
    elif abs(s1 - s2) < tol:
        if s0 < s1: w[0] = w[1] = 0.5
        else:       w[3] = w[5] = 0.5
    elif abs(s0 - s2) < tol:
        if s0 < s1: w[1] = w[4] = 0.5
        else:       w[2] = w[3] = 0.5
    else:
        w[indice_patron(seg)] = 1.0
    return w


def _normalizar_filas(m: np.ndarray) -> np.ndarray:
    out = np.zeros_like(m)
    rs = m.sum(axis=1)
    for i in range(m.shape[0]):
        if rs[i] > 0:
            out[i] = m[i] / rs[i]
    return out


def matrices_optm(x: np.ndarray) -> dict[str, np.ndarray]:
    """OPTM con las reglas 'omision', 'secuencial' y 'ponderado' (filas normalizadas)."""
    x = np.asarray(x, dtype=float)
    pat_s, pat_0, pesos = [], [], []
    for t in range(len(x) - 2):
        seg = [x[t], x[t + 1], x[t + 2]]
        pat_s.append(indice_patron(seg))
        pesos.append(_pesos_ponderados(seg))
        if len(set(seg)) == 3:
            pat_0.append(indice_patron(seg))
    ms = np.zeros((6, 6)); m0 = np.zeros((6, 6)); mw = np.zeros((6, 6))
    for k in range(len(pat_s) - 1):
        ms[pat_s[k], pat_s[k + 1]] += 1
        mw += np.outer(pesos[k], pesos[k + 1])
    for k in range(len(pat_0) - 1):
        m0[pat_0[k], pat_0[k + 1]] += 1
    return {"omision": _normalizar_filas(m0), "secuencial": _normalizar_filas(ms),
            "ponderado": _normalizar_filas(mw)}


def vector_estacionario(m: np.ndarray) -> np.ndarray:
    vals, vecs = eig(m.T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
    return v / v.sum() if v.sum() != 0 else v


def norma_estacionaria(x: np.ndarray, regla: str = "secuencial") -> float:
    """‖v‖ del vector estacionario de la OPTM (Stat_MS para la regla secuencial)."""
    return float(np.linalg.norm(vector_estacionario(matrices_optm(x)[regla])))


def clasificar_norma(v: float, regla: str = "secuencial") -> str:
    u = UMBRALES[regla]
    return "NDBT2" if v <= u["sN"] else ("DBT2" if v > u["sP"] else "dudoso")


def clasificar(x: np.ndarray) -> dict:
    """Clasificadores ℓ1 (H×C), ℓF (Frobenius) y ℓv (‖v‖) e índice compuesto ν.

    ν = min( √(2·A(ℓ1)² + A(ℓF)² + A(ℓv)²), 1 ), con A(N)=0, A(D)=0,25, A(P)=0,5.
    """
    mats = matrices_optm(x)
    h, c = hxc(x)
    dn = float(np.hypot(h - CENTROIDE_N[0], c - CENTROIDE_N[1]))
    dp = float(np.hypot(h - CENTROIDE_P[0], c - CENTROIDE_P[1]))
    l1 = "N" if dn < min(dp, ETA) else ("P" if dp < min(dn, ETA) else "D")
    a = {"N": 0.0, "D": 0.25, "P": 0.5}
    out = {}
    for regla, m in mats.items():
        u = UMBRALES[regla]
        f = float(np.linalg.norm(m, "fro"))
        sv = float(np.linalg.norm(vector_estacionario(m)))
        lf = "N" if f <= u["fN"] else ("P" if f > u["fP"] else "D")
        lv = "N" if sv <= u["sN"] else ("P" if sv > u["sP"] else "D")
        nu = min(np.sqrt(2 * a[l1] ** 2 + a[lf] ** 2 + a[lv] ** 2), 1.0)
        out[regla] = dict(H=h, C=c, frobenius=f, norma_v=sv, l1=l1, lF=lf, lv=lv, nu=float(nu))
    return out
