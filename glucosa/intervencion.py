"""Intervención simulada sobre el forzamiento S(t) — Capítulos 6 y 7.

* Descomposición wavelet no decimada «à trous» (starlet, núcleo B3-spline),
  escalas 1–5 (≈ 0,5–8 h)                                            (6.1–6.3)
* Energía wavelet por evento                                         (6.4)
* Ventana con rampas cosenoidales de r = 1 h                         (6.5a–b)
* Operador de atenuación posprandial S → S′                          (6.5c–e)
* Nivel de referencia por bloques de 3 días con actualización causal (6.6a–b)
* Extensión rápida de seguridad ante hipoglucemia S′ → S_corr        (6.7a–j)
* HbA1c equivalente ADAG y GMI                                       (6.8)
* Optimización de las fracciones de energía ρ_i con Optuna/TPE       (6.9–6.10)

Las constantes de cómputo se declaran en `PESOS_COSTO` y en el Anexo B.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.signal import savgol_filter

from . import sdde
from .forzamiento import forzamiento
from .datos import MUESTRAS_DIA

H_B3 = np.array([1, 4, 6, 4, 1], dtype=float) / 16.0
ESCALAS = 5
RAMPA = 4                     # r = 1 h = 4 muestras de 15 min
DIAS_BLOQUE = 3

#: Pesos de la función de costo (6.10) (Anexo B).
PESOS_COSTO = dict(w_A=10.0, w_TBR=100.0, w_e=1.0, tau_TBR=0.02)

#: Parámetros de la extensión de seguridad (Tabla 7.8).
SEGURIDAD = dict(k_d=0.30, k_v=0.02, gamma_n=1.00, S_h_max=0.75,
                 tau_ataque=10.0, tau_retorno=60.0)


# --------------------------------------------------------------------------- indicadores
def a1c_adag(g_media: float) -> float:
    """HbA1c equivalente (%) = (Ḡ + 46,7)/28,7 (Nathan et al., 2008)."""
    return (g_media + 46.7) / 28.7


def gmi(g_media: float) -> float:
    """Glucose Management Indicator (%) = 3,31 + 0,02392·Ḡ (Bergenstal et al., 2018)."""
    return 3.31 + 0.02392 * g_media


def episodio_max_bajo(g: np.ndarray, umbral: float = 70.0, dt_h: float = 0.25) -> float:
    """Duración [h] del episodio consecutivo más largo con G < umbral."""
    mejor = act = 0
    for v in np.asarray(g) < umbral:
        act = act + 1 if v else 0
        mejor = max(mejor, act)
    return mejor * dt_h


def resumen(g: np.ndarray) -> dict:
    g = np.asarray(g, float)
    m = g.mean()
    return dict(G_media=m, A1c=a1c_adag(m), GMI=gmi(m),
                TIR=100 * np.mean((g >= 70) & (g <= 180)), TAR=100 * np.mean(g > 180),
                TBR=100 * np.mean(g < 70), TBR54=100 * np.mean(g < 54),
                hipo_max_h=episodio_max_bajo(g))


# --------------------------------------------------------------------------- starlet
def starlet(s: np.ndarray, J: int = ESCALAS):
    """Detalles w_1…w_J y aproximación c_J (borde por reflexión)."""
    c = np.asarray(s, dtype=float).copy()
    n = len(c)
    detalles = []
    for j in range(1, J + 1):
        paso = 2 ** (j - 1)
        idx = np.arange(n)
        nuevo = np.zeros(n)
        for k, h in zip(range(-2, 3), H_B3):
            p = idx + k * paso
            p = np.abs(p)                                   # reflexión izquierda
            p = np.where(p >= n, 2 * (n - 1) - p, p)        # reflexión derecha
            nuevo += h * c[p]
        detalles.append(c - nuevo)                         # (6.2)
        c = nuevo                                          # (6.1)
    return detalles, c


def componente_posprandial(s: np.ndarray, J: int = ESCALAS) -> tuple[np.ndarray, list]:
    detalles, _ = starlet(s, J)
    return np.sum(detalles, axis=0), detalles


def energia_eventos(detalles, eventos) -> np.ndarray:
    """E_i = Σ_j Σ_{t∈I_i} w_j²(t)   (6.4)."""
    w2 = np.sum(np.square(detalles), axis=0)
    return np.array([w2[a:b + 1].sum() for a, b in eventos])


def ventana(n: int, a: int, b: int, r: int = RAMPA) -> np.ndarray:
    """m_i(t) con rampas cosenoidales (6.5a–b)."""
    t = np.arange(n, dtype=float)
    m = np.zeros(n)
    m[(t >= a) & (t <= b)] = 1.0
    sub = (t >= a - r) & (t < a)
    m[sub] = 0.5 * (1 - np.cos(np.pi * (t[sub] - a + r) / r))
    baj = (t > b) & (t <= b + r)
    m[baj] = 0.5 * (1 + np.cos(np.pi * (t[baj] - b) / r))
    return m


def atenuar(s: np.ndarray, eventos, rho, s_b: np.ndarray | None = None) -> np.ndarray:
    """S′(t) = S(t) + Σ_i (√ρ_i − 1) m_i(t) S_B(t)   (6.5c–e)."""
    s = np.asarray(s, float)
    s_b = componente_posprandial(s)[0] if s_b is None else s_b
    out = s.copy()
    for (a, b), r in zip(eventos, rho):
        out += (np.sqrt(r) - 1.0) * ventana(len(s), a, b) * s_b
    return out


# --------------------------------------------------------------------------- re-simulación
def simular_intervencion(g_obs, p: dict, s_act: np.ndarray, causal: bool = True,
                         seguridad: dict | None = None, ruido=None, g_min: float = sdde.G_MIN,
                         g_ref: np.ndarray | None = None):
    """Re-integra la SDDE con forzamiento ``s_act`` y referencia causal por bloques (6.6b).

    El primer bloque usa el nivel de la referencia (6.6a). Al completar cada bloque I_k,
    el nivel del bloque siguiente se desplaza en la diferencia entre la media alcanzada
    por la trayectoria intervenida y la de la trayectoria de referencia en I_k:

        B_int(t ∈ I_{k+1}) = B(t ∈ I_{k+1}) + ⟨G_int⟩_{I_k} − ⟨G_ref⟩_{I_k}

    Así, sin intervención (ρ_i = 1) se recupera exactamente la trayectoria de referencia y
    toda diferencia entre ambas proviene de la intervención. ``g_ref`` es la trayectoria de
    referencia con la misma realización de ruido (se calcula si no se pasa).
    Con ``seguridad`` se aplica en lazo cerrado la corrección (6.7a–j).
    """
    g_obs = np.asarray(g_obs, float)
    n = len(g_obs)
    b_ref = sdde.referencia_por_bloques(g_obs)
    ruido = sdde.ruido_blanco(n) if ruido is None else ruido
    if not causal and seguridad is None:
        return sdde.simular(g_obs, s=s_act, b=b_ref, ruido=ruido, g_min=g_min, **p)
    if causal and g_ref is None:
        g_ref = sdde.simular(g_obs, b=b_ref, ruido=ruido, g_min=g_min, **p)
    tam = MUESTRAS_DIA * DIAS_BLOQUE
    b = b_ref.copy()
    g = None
    for k0 in range(0, n, tam):
        g = _simular_con_seguridad(g_obs, p, s_act, b, ruido, seguridad, g_min) \
            if seguridad is not None else sdde.simular(g_obs, s=s_act, b=b, ruido=ruido, g_min=g_min, **p)
        if causal and k0 + tam < n:
            b[k0 + tam:k0 + 2 * tam] = b_ref[k0 + tam:k0 + 2 * tam] + g[k0:k0 + tam].mean() - g_ref[k0:k0 + tam].mean()
    return g


@njit(cache=True)
def _integrar_seguridad(s_f, alpha, pasos_tau, k_f, beta_n, b_f, delta_b, g0, g_min, g_max,
                        ruido, dt, f_max, k_d, k_v, gamma_n, sh_max, t_at, t_ret):
    n = len(s_f)
    g = np.zeros(n)
    sh_hist = np.zeros(n)
    g[:pasos_tau + 1] = g0
    sh = 0.0
    raiz = np.sqrt(dt)
    for t in range(pasos_tau, n - 1):
        sp = 0.5 * (s_f[t] + abs(s_f[t]))                     # (6.7a)
        sn = 0.5 * (-s_f[t] + abs(s_f[t]))
        d = (70.0 - g[t] + abs(70.0 - g[t])) / 32.0           # (6.7c)
        c = min(1.0, max(0.0, (78.0 - g[t]) / 8.0))           # (6.7d)
        gp = (g[t] - g[t - 1]) / dt if t > 0 else 0.0
        a = c * 0.5 * (-gp + abs(gp))                         # (6.7e)
        obj = min(sh_max, k_d * d + k_v * a)                  # (6.7f)
        tau_h = t_at if obj > sh else t_ret                   # (6.7h)
        sh += dt * (obj - sh) / tau_h                         # (6.7g)
        ah = min(1.0, max(0.0, sh / sh_max))                  # (6.7i)
        rho_n = 1.0 - gamma_n * ah                            # (6.7j)
        s_corr = sp - rho_n * sn + sh                         # (6.7b)
        sh_hist[t] = sh
        f = -alpha * (g[t - pasos_tau] - (b_f[t] + delta_b)) + k_f * s_corr
        if f > f_max:
            f = f_max
        if f < -f_max:
            f = -f_max
        g[t + 1] = g[t] + f * dt + beta_n * ruido[t] * raiz
        if g[t + 1] < g_min:
            g[t + 1] = g_min
        if g[t + 1] > g_max:
            g[t + 1] = g_max
    return g, sh_hist


def _simular_con_seguridad(g_obs, p, s_act, b, ruido, seg, g_min):
    n = len(g_obs)
    r = sdde.RAZON
    s_f = np.repeat(np.asarray(s_act, float), r)[:n * r]
    b_f = np.repeat(np.asarray(b, float), r)[:n * r]
    fina, _ = _integrar_seguridad(s_f, p["alpha"], int(p["tau_d"] / sdde.DT_SIM), p["k_f"],
                                  p["beta_n"], b_f, p["delta_b"], g_obs[0], g_min, sdde.G_MAX,
                                  ruido, sdde.DT_SIM, sdde.F_MAX, seg["k_d"], seg["k_v"],
                                  seg["gamma_n"], seg["S_h_max"], seg["tau_ataque"],
                                  seg["tau_retorno"])
    if p.get("filtro_sg", True):
        suave = savgol_filter(fina, sdde.VENTANA_SG, 3, mode="nearest")
        suave[0] = fina[0]
        fina = suave
    return fina[::r][:n]


# --------------------------------------------------------------------------- optimización
@dataclass
class ResultadoOptimizacion:
    rho: np.ndarray
    eventos: list
    energia_ref: np.ndarray
    energia_final: np.ndarray
    g_referencia: np.ndarray
    g_intervenida: np.ndarray
    s_intervenido: np.ndarray
    historial: np.ndarray


def costo_intervencion(g: np.ndarray, rho, a1c_obj: float, pesos=PESOS_COSTO) -> float:
    """J(ρ) = w_A·max(0, A1c − A1c_obj) + w_TBR·max(0, TBR − τ_TBR) + w_e·mean|1 − ρ|  (6.10)."""
    a1c = a1c_adag(np.mean(g))
    tbr = np.mean(np.asarray(g) < 70)
    reg = np.mean(np.abs(1 - np.asarray(rho))) if len(rho) else 0.0
    return (pesos["w_A"] * max(0.0, a1c - a1c_obj) + pesos["w_TBR"] * max(0.0, tbr - pesos["tau_TBR"])
            + pesos["w_e"] * reg)


def optimizar(g_obs, p: dict, eventos, a1c_obj: float = 6.8, n_ensayos: int = 250,
              semilla: int = 42, pesos=PESOS_COSTO, g_referencia=None,
              g_min: float = sdde.G_MIN, causal: bool = True) -> ResultadoOptimizacion:
    """TPE sobre las fracciones de energía ρ_i. ``causal=False`` re-integra con la referencia
    por bloques de la señal medida (6.6a) en lugar de la actualización causal (6.6b)."""
    import optuna
    g_obs = np.asarray(g_obs, float)
    s = forzamiento(g_obs)
    s_b, detalles = componente_posprandial(s)
    ruido = sdde.ruido_blanco(len(g_obs))
    g_ref = sdde.simular(g_obs, ruido=ruido, g_min=g_min, **p) if g_referencia is None else g_referencia

    def objetivo(trial):
        rho = [trial.suggest_float(f"rho_{i}", 0.0, 1.0) for i in range(len(eventos))]
        g = simular_intervencion(g_obs, p, atenuar(s, eventos, rho, s_b), ruido=ruido, g_min=g_min, causal=causal, g_ref=g_ref)
        return costo_intervencion(g, rho, a1c_obj, pesos)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    est = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=semilla))
    if len(eventos):
        est.enqueue_trial({f"rho_{i}": 1.0 for i in range(len(eventos))})   # punto de partida: sin intervención
        est.optimize(objetivo, n_trials=n_ensayos)
        rho = np.array([est.best_params[f"rho_{i}"] for i in range(len(eventos))])
        hist = np.minimum.accumulate([t.value for t in est.trials])
    else:
        rho, hist = np.array([]), np.array([])
    s_int = atenuar(s, eventos, rho, s_b)
    g_int = simular_intervencion(g_obs, p, s_int, ruido=ruido, g_min=g_min, causal=causal, g_ref=g_ref)
    e_ref = energia_eventos(detalles, eventos)
    e_fin = e_ref * rho
    return ResultadoOptimizacion(rho, list(eventos), e_ref, e_fin, g_ref, g_int, s_int, hist)
