"""Cohorte pública CGMacros (Das et al., 2025; Gutierrez-Osuna et al., 2025) — Capítulo 7.

CGMacros se distribuye en PhysioNet bajo licencia CC BY-NC-SA 4.0
(https://doi.org/10.13026/3z8q-x658). El repositorio incluye una versión compacta
(``datos/cgmacros/compacto``: lecturas Libre de 1 min, marcas de comidas y HbA1c),
adaptación bajo la misma licencia. Los originales pueden descargarse con
``scripts/descargar_cgmacros.py``; ambas fuentes producen las mismas series.

Preprocesamiento (Sec. 7.2 y Anexo B.1)
---------------------------------------
* Señal primaria: ``Libre GL``. El archivo público está en una grilla de 1 min
  interpolada por los autores; se remuestrea a 15 min promediando las lecturas de cada
  intervalo de reloj (esta elección reproduce exactamente la Tabla 7.3).
* Instantes sin fila en el archivo → faltantes. Huecos internos de hasta 4
  muestras: interpolación lineal. Huecos mayores: reconstrucción condicionada
  DCT + PCA (base: perfiles diarios completos de los sujetos sanos; 32
  coeficientes DCT; 22 componentes principales), sólo si el registro conserva
  al menos 6 días utilizables (muestras por encima del piso del sensor, 40 mg/dL).
  No se extrapolan los extremos.
* Grupo según HbA1c de laboratorio (ADA): < 5,7 % sano; 5,7–6,4 % prediabetes;
  ≥ 6,5 % diabetes tipo 2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fftpack import dct, idct

from .datos import DATOS

CARPETA = DATOS / "cgmacros" / "CGMacros"
COMPACTO = DATOS / "cgmacros" / "compacto"
_CACHE: dict = {}
PISO_SENSOR = 40.0
MAX_INTERPOLAR = 4
DIAS_MINIMOS = 6.0
N_DCT, N_PCA = 32, 22


def grupo_por_hba1c(a1c: float) -> str:
    if a1c < 5.7:
        return "Sano"
    if a1c < 6.5:
        return "Prediabético"
    return "Diabetes tipo 2"


def ids_disponibles() -> list[int]:
    if (COMPACTO / "bio.csv").exists():
        return sorted(pd.read_csv(COMPACTO / "libre_1min.csv.gz", usecols=["sujeto"]).sujeto.unique().tolist()) \
            if "ids" not in _CACHE else _CACHE["ids"]
    return sorted(int(p.name.split("-")[1]) for p in CARPETA.glob("CGMacros-0*"))


def cargar_bio(carpeta: Path = CARPETA) -> pd.DataFrame:
    ruta = COMPACTO / "bio.csv" if (COMPACTO / "bio.csv").exists() else carpeta / "bio.csv"
    bio = pd.read_csv(ruta)
    bio.columns = [c.strip() for c in bio.columns]
    bio["grupo"] = bio["A1c PDL (Lab)"].apply(grupo_por_hba1c)
    return bio


@dataclass
class Registro:
    sujeto: int
    grupo: str
    tiempo: pd.DatetimeIndex
    glucosa: np.ndarray                 # 15 min, con NaN en huecos > 4 muestras
    faltante: np.ndarray                # máscara de huecos originales
    comidas: pd.DatetimeIndex
    dias_utilizables: float
    huecos_largos: list = field(default_factory=list)   # (inicio, largo) en muestras
    reconstruido: bool = False

    @property
    def desfase(self) -> int:
        """Hora del día de la primera muestra, en intervalos de 15 min."""
        t0 = self.tiempo[0]
        return (t0.hour * 60 + t0.minute) // 15

    @property
    def completo(self) -> bool:
        return not np.isnan(self.glucosa).any()


def cargar_registro(sujeto: int, bio: pd.DataFrame | None = None, carpeta: Path = CARPETA) -> Registro:
    bio = cargar_bio(carpeta) if bio is None else bio
    if (COMPACTO / "libre_1min.csv.gz").exists():
        if "libre" not in _CACHE:
            _CACHE["libre"] = pd.read_csv(COMPACTO / "libre_1min.csv.gz", parse_dates=["timestamp"])
            _CACHE["comidas"] = pd.read_csv(COMPACTO / "comidas.csv", parse_dates=["timestamp"])
        x = _CACHE["libre"][_CACHE["libre"].sujeto == sujeto]
        ts = pd.DatetimeIndex(x.timestamp)
        s = pd.Series(x.libre_gl.to_numpy(dtype=float), index=ts)
        c = _CACHE["comidas"]
        comidas = pd.DatetimeIndex(c[c.sujeto == sujeto].timestamp)
    else:
        d = pd.read_csv(carpeta / f"CGMacros-{sujeto:03d}" / f"CGMacros-{sujeto:03d}.csv", low_memory=False)
        d.columns = [c.strip() for c in d.columns]
        ts = pd.to_datetime(d["Timestamp"])
        s = pd.Series(pd.to_numeric(d["Libre GL"], errors="coerce").to_numpy(), index=ts)
        comidas = pd.DatetimeIndex(ts[d["Meal Type"].notna() & (d["Meal Type"].astype(str).str.strip() != "")])
    s = s[~s.index.duplicated()].dropna()
    # Remuestreo a 15 min: media de las lecturas de 1 min en cada intervalo de reloj
    # [hh:00, hh:15), [hh:15, hh:30), … ; los intervalos sin lecturas quedan como faltantes.
    r15 = s.resample("15min").mean()
    grilla = r15.index
    g = r15.to_numpy(dtype=float).copy()
    # Lecturas aisladas (un único intervalo observado entre dos huecos de más de 1 h, como
    # en el sujeto 018) se tratan como parte del hueco.
    obs = ~np.isnan(g)
    for k in np.where(obs)[0]:
        izq = g[max(0, k - 5):k]; der = g[k + 1:k + 6]
        if len(izq) == 5 and len(der) == 5 and np.isnan(izq).all() and np.isnan(der).all():
            g[k] = np.nan
    falt = np.isnan(g)
    # huecos internos
    largos, k = [], 0
    while k < len(g):
        if falt[k]:
            j = k
            while j < len(g) and falt[j]:
                j += 1
            largos.append((k, j - k))
            k = j
        else:
            k += 1
    g_i = pd.Series(g).interpolate(limit_area="inside").to_numpy().copy()
    for a, n in largos:
        if n > MAX_INTERPOLAR:
            g_i[a:a + n] = np.nan
    a1c = float(bio.loc[bio.subject == sujeto, "A1c PDL (Lab)"].iloc[0])
    util = (len(g_i) - np.sum(g_i <= PISO_SENSOR)) / 96.0          # días no fijados en el piso del sensor
    return Registro(sujeto, grupo_por_hba1c(a1c), grilla, g_i, falt, comidas, float(util),
                    [(a, n) for a, n in largos if n > MAX_INTERPOLAR])


# --------------------------------------------------------------------------- DCT + PCA
class BaseDCTPCA:
    """Perfiles diarios (96 muestras) → DCT (32 coef.) → PCA (22 componentes)."""

    def __init__(self, perfiles: np.ndarray, n_dct: int = N_DCT, n_pca: int = N_PCA):
        self.n_dct, self.n_pca = n_dct, n_pca
        X = dct(perfiles, type=2, norm="ortho", axis=1)[:, :n_dct]
        self.media = X.mean(0)
        u, s, vt = np.linalg.svd(X - self.media, full_matrices=False)
        self.componentes = vt[:n_pca]                       # (n_pca, n_dct)
        self.varianzas = (s[:n_pca] ** 2) / (len(X) - 1)
        self.var_explicada = float(np.sum(s[:n_pca] ** 2) / np.sum(s ** 2))
        self.n_perfiles = len(perfiles)
        # matriz que lleva puntajes PCA → perfil temporal
        base_t = idct(np.eye(96)[:, :n_dct], type=2, norm="ortho", axis=0)   # (96, n_dct)
        self.B = base_t @ self.componentes.T                                     # (96, n_pca)
        self.mu_t = base_t @ self.media
        resid = perfiles - (self.mu_t + (perfiles - self.mu_t) @ self.B @ np.linalg.pinv(self.B))
        self.sigma_ruido = float(np.std(resid))

    #: Fracción del desvío residual (no explicado por las 22 componentes) que se agrega
    #: como ruido blanco a cada realización (Anexo B).
    fraccion_ruido: float = 0.35

    def condicionar(self, dia: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Una realización del día completo condicionada a las muestras observadas."""
        obs = ~np.isnan(dia)
        P = np.diag(1 / self.varianzas)
        if obs.sum():
            Bo = self.B[obs]
            A = Bo.T @ Bo / self.sigma_ruido ** 2 + P
            cov = np.linalg.inv(A)
            m = cov @ Bo.T @ (dia[obs] - self.mu_t[obs]) / self.sigma_ruido ** 2
        else:
            cov, m = np.diag(self.varianzas), np.zeros(self.n_pca)
        z = rng.multivariate_normal(m, cov)
        perfil = self.mu_t + self.B @ z + rng.normal(0, self.fraccion_ruido * self.sigma_ruido, 96)
        out = dia.copy()
        out[~obs] = perfil[~obs]
        return out


def reconstruir(reg: Registro, base: BaseDCTPCA, n_realizaciones: int = 100, semilla: int = 42):
    """100 realizaciones de la serie con los huecos largos completados; devuelve (realizaciones, elegida)."""
    from .ordinal import hxc
    rng = np.random.default_rng(semilla)
    g = reg.glucosa
    k0 = reg.desfase
    n_dias = int(np.ceil((len(g) + k0) / 96))
    pad = np.full(n_dias * 96, np.nan)
    pad[k0:k0 + len(g)] = g
    reales = []
    for _ in range(n_realizaciones):
        y = pad.copy()
        for d in range(n_dias):
            dia = y[d * 96:(d + 1) * 96]
            if np.isnan(dia).any():
                y[d * 96:(d + 1) * 96] = base.condicionar(dia, rng)
        y = y[k0:k0 + len(g)]
        y = np.where(np.isnan(g), np.clip(y, PISO_SENSOR, 400), g)
        reales.append(y)
    reales = np.array(reales)
    hc = np.array([hxc(r) for r in reales])
    med = np.median(hc, axis=0)
    k = int(np.argmin(np.hypot(*(hc - med).T)))
    return reales, k, hc


def perfiles_diarios_sanos(registros: list[Registro]) -> np.ndarray:
    """Días calendario completos (00:00–23:45) de los sujetos sanos."""
    perf = []
    for r in registros:
        if r.grupo != "Sano":
            continue
        g = r.glucosa
        k0 = (96 - r.desfase) % 96
        for d in range((len(g) - k0) // 96):
            dia = g[k0 + d * 96:k0 + (d + 1) * 96]
            if not np.isnan(dia).any() and np.all(dia > PISO_SENSOR):
                perf.append(dia)
    return np.array(perf)
