"""Detección de eventos compatibles con fenómeno del alba e ingesta — Cap. 3, Sec. 3.3.6.

Índice de predominio positivo   I(y;J) = (A⁺ − A⁻)/(A⁺ + A⁻ + ε)          (3.17)
Energía wavelet de banda        W_S(t) = mean_{T∈banda} |C_S(t,T)|²       (3.29, 3.32)

Fenómeno del alba (3.3.6.2): ventana diaria 03:00–08:00 h, exclusión si el
mínimo de G en las 5 h previas es < 70 mg/dL, banda lenta 3–8 h.

Ingesta (3.3.6.3–3.3.6.4): ventanas de L = 2,5 h con inicios 08:00–21:30 h cada
15 min, ΔG_max ≥ 20 mg/dL, puntaje Q_k (3.35), selección voraz sin solapes,
descarte Q_k < 0,2 y rótulos fuerte / moderada / posible.

Parámetros de implementación (Anexo B):
* núcleo de Morlet complejo ``cmor1.0-1.0`` de PyWavelets, 24 períodos
  log-espaciados en la banda;
* W_S normalizada por su máximo en el registro (así W ≤ 1 y el umbral 0,15 de
  la categoría «fuerte» es una fracción del máximo);
* ε_S = ε_G = 0 en (3.35);
* ventanas cerradas [n_a, n_b] (inclusive).
* Gradación del alba (Tabla 3.3, descrita como apoyo visual): fuerte si
  I_G > 0,5, I_S > 0,5 y W̄ ≥ mediana de las ventanas del sujeto; posible si
  min(I_G, I_S) < 0,15; moderada en el resto de las compatibles.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pywt

DT_H = 0.25
MUESTRAS_DIA = 96
BANDA_ALBA = (3.0, 8.0)
BANDA_INGESTA = (0.5, 2.0)
WAVELET = "cmor1.0-1.0"


def energia_banda(s: np.ndarray, banda_h, wavelet: str = WAVELET, n_periodos: int = 24,
                  normalizar: bool = True) -> np.ndarray:
    """W_S(t): media de |C_S(t,T)|² sobre los períodos T de la banda (en horas)."""
    s = np.asarray(s, dtype=float)
    periodos = np.geomspace(banda_h[0], banda_h[1], n_periodos)
    escalas = pywt.central_frequency(wavelet) * periodos / DT_H
    coef, _ = pywt.cwt(s, escalas, wavelet, sampling_period=DT_H)
    w = np.mean(np.abs(coef) ** 2, axis=0)
    return w / w.max() if normalizar and w.max() > 0 else w


def indice_predominio(y: np.ndarray, eps: float = 1e-9) -> float:
    y = np.asarray(y, dtype=float)
    ap = np.sum(np.maximum(y, 0))
    an = np.sum(np.maximum(-y, 0))
    return float((ap - an) / (ap + an + eps))


# --------------------------------------------------------------------------- alba
@dataclass
class VentanaAlba:
    dia: int
    inicio: int
    fin: int
    delta_g: float
    I_G: float
    I_S: float
    W_media: float
    minimo_previo: float
    excluida_hipo: bool
    compatible: bool
    categoria: str = ""


def detectar_alba(g: np.ndarray, s: np.ndarray, umbral_hipo: float = 70.0,
                  desfase: int = 0, dg_min: float = 0.0) -> list[VentanaAlba]:
    """Ventanas 03:00–08:00 h. ``desfase``: hora del día de la primera muestra, en
    muestras de 15 min (0 si el registro empieza a las 00:00). ``dg_min``: umbral de
    magnitud adicional (30 mg/dL en el Capítulo 7)."""
    g = np.asarray(g, float)
    w = energia_banda(s, BANDA_ALBA)
    out = []
    for d in range(-1, (len(g) + desfase) // MUESTRAS_DIA + 1):
        a, b = d * MUESTRAS_DIA + 12 - desfase, d * MUESTRAS_DIA + 32 - desfase   # 03:00 y 08:00
        if a < 0 or b >= len(g):
            continue
        seg = g[a:b + 1]
        previo = g[max(0, a - 20):a]
        mprev = float(previo.min()) if len(previo) else np.nan
        excl = bool(len(previo) and mprev < umbral_hipo)
        dg = float(g[b] - g[a])
        ig = indice_predominio(seg - seg[0])
        is_ = indice_predominio(s[a:b + 1])
        comp = (dg > 0) and (dg >= dg_min) and (ig > 0) and (is_ > 0) and not excl
        out.append(VentanaAlba(d + 1, a, b, dg, ig, is_, float(w[a:b + 1].mean()), mprev, excl, comp))
    med = np.median([v.W_media for v in out])
    for v in out:
        if not v.compatible:
            v.categoria = "posible" if (v.delta_g > 0 and not v.excluida_hipo) else "no compatible"
        elif v.I_G > 0.5 and v.I_S > 0.5 and v.W_media >= med:
            v.categoria = "fuerte"
        elif min(v.I_G, v.I_S) < 0.15:
            v.categoria = "posible"
        else:
            v.categoria = "moderada"
    return out


# --------------------------------------------------------------------------- ingesta
@dataclass
class VentanaIngesta:
    inicio: int
    fin: int
    Q: float
    delta_g_max: float
    I_G: float
    I_S: float
    W_max: float
    categoria: str = ""


def detectar_ingestas(g: np.ndarray, s: np.ndarray, largo: int = 10, dg_min: float = 20.0,
                      q_min: float = 0.2, eps_s: float = 0.0, eps_g: float = 0.0,
                      hora_inicio: float = 8.0, hora_fin: float = 21.5,
                      w: np.ndarray | None = None, desfase: int = 0) -> list[VentanaIngesta]:
    """Ventanas no solapadas compatibles con ingesta (L = 2,5 h = 10 intervalos).

    ``desfase``: hora del día de la primera muestra, en muestras de 15 min.
    """
    g = np.asarray(g, float)
    s = np.asarray(s, float)
    w = energia_banda(s, BANDA_INGESTA) if w is None else w
    k0, k1 = int(hora_inicio * 4), int(hora_fin * 4)
    cand = []
    for d in range(-1, int(np.ceil((len(g) + desfase) / MUESTRAS_DIA)) + 1):
        for k in range(k0, k1 + 1):
            a = d * MUESTRAS_DIA + k - desfase
            b = a + largo
            if a < 0 or b >= len(g):
                continue
            seg = g[a:b + 1]
            dg = float(np.max(seg - seg[0]))
            if dg < dg_min:
                continue
            ig = indice_predominio(seg - seg[0])
            is_ = indice_predominio(s[a:b + 1])
            wm = float(np.max(w[a:b + 1]))
            q = dg / 40 * (max(is_, 0) + eps_s) * (1 + min(wm, 1.5)) * (max(ig, 0) + eps_g)   # (3.35)
            cand.append(VentanaIngesta(a, b, q, dg, ig, is_, wm))
    cand.sort(key=lambda v: -v.Q)
    elegidas: list[VentanaIngesta] = []
    for v in cand:
        if v.Q < q_min:
            continue
        if all(v.fin < o.inicio or o.fin < v.inicio for o in elegidas):            # (3.35a)
            elegidas.append(v)
    for v in elegidas:
        if v.delta_g_max >= 120 and v.I_G > 0.70 and v.I_S > 0.65 and v.W_max > 0.15:
            v.categoria = "fuerte"
        elif v.delta_g_max >= 70 and v.I_G > 0.55 and v.I_S > 0.35:
            v.categoria = "moderada"
        elif v.delta_g_max >= 25 and v.I_G > 0 and v.I_S > 0:
            v.categoria = "posible"
        else:
            v.categoria = "no compatible"
    return sorted(elegidas, key=lambda v: v.inicio)
