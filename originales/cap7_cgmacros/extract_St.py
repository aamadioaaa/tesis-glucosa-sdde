from _rutas import _CGM, _SAL  # rutas portables (única línea agregada al publicar)
"""
Extracción de S(t) — Indicador de Forzamiento Metabólico (Capítulo 3, ec. 3.6–3.13)
Dataset: CGMacros. Fuente G(t): Libre GL. Muestreo de trabajo: 15 min.
"""
import pandas as pd, numpy as np, glob, os
from scipy.fftpack import dct, idct

BASE = _CGM
OUT  = _SAL+"/S_t"
os.makedirs(OUT, exist_ok=True)

# ---- Parámetros (Capítulo 3) ----
DT_MIN     = 15      # intervalo de muestreo de trabajo (ec. 3.6)
GAMMA      = 0.5     # factor de compresión tanh (ec. 3.7)
WIN        = 10      # ventana E_local: 10 muestras = 150 min (sec. 3.2.2)
KC_FRAC    = 0.10    # corte DCT: conservar 10% de coef. de baja frec. (k_c; valor por defecto, texto en blanco)
SRC_COL    = "Libre GL"

def dct_lowpass(x, kc_frac):
    N = len(x)
    C = dct(x, type=2, norm='ortho')
    kc = max(1, int(np.ceil(kc_frac * N)))
    C[kc:] = 0.0
    return idct(C, type=2, norm='ortho')

def process(subj_dir):
    subj = os.path.basename(subj_dir)
    f = os.path.join(subj_dir, f"{subj}.csv")
    if not os.path.exists(f): return None
    df = pd.read_csv(f)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    s = df.set_index('Timestamp')[SRC_COL].sort_index()
    # Remuestreo a 15 min (media dentro del bin) e interpolación de huecos
    s15 = s.resample(f'{DT_MIN}min').mean().interpolate('time', limit_direction='both')
    G = s15.values.astype(float)
    t = s15.index
    if len(G) < 20: return None

    # v(t): diferencias centradas (ec. 3.6), unidades mg/dL/min
    v = np.gradient(G, DT_MIN)
    sigma_v = np.std(v, ddof=0)
    if sigma_v == 0: sigma_v = 1e-9
    # velocidad acondicionada (ec. 3.7)
    v_cond = sigma_v * np.tanh(v / (sigma_v * GAMMA))

    # E_local: varianza en ventana deslizante centrada de 10 muestras (ec. 3.8)
    E_local = pd.Series(G, index=t).rolling(WIN, center=True, min_periods=1).var(ddof=0).values
    E_local = np.nan_to_num(E_local, nan=0.0)
    # Suavizado DCT pasa-bajos (ec. 3.9-3.11)
    E_smooth = dct_lowpass(E_local, KC_FRAC)
    E_smooth = np.clip(E_smooth, 0, None)   # varianza no negativa
    # Normalización por máximo histórico del sujeto (ec. 3.12)
    Emax = E_smooth.max()
    E_hat = E_smooth / Emax if Emax > 0 else np.zeros_like(E_smooth)

    # S(t) — indicador de forzamiento metabólico (ec. 3.13)
    S = v_cond * (0.2 + 0.8 * E_hat)

    out = pd.DataFrame({
        'Timestamp': t, 'G': G, 'v': v, 'v_cond': v_cond,
        'E_local': E_local, 'E_hat': E_hat, 'S': S
    })
    out.to_csv(os.path.join(OUT, f"{subj}_St.csv"), index=False)
    return dict(subject=subj, n=len(out), days=round((t[-1]-t[0]).total_seconds()/86400,2),
                sigma_v=round(sigma_v,4), S_mean=round(S.mean(),4), S_std=round(S.std(),4),
                S_min=round(S.min(),3), S_max=round(S.max(),3))

rows=[]
for d in sorted(glob.glob(os.path.join(BASE,"CGMacros-*"))):
    r = process(d)
    if r: rows.append(r)

summary = pd.DataFrame(rows)
summary.to_csv(_SAL+"/S_t_resumen.csv", index=False)
# consolidado
allparts=[]
for d in sorted(glob.glob(os.path.join(OUT,"*_St.csv"))):
    p=pd.read_csv(d); p.insert(0,'subject',os.path.basename(d).replace('_St.csv',''))
    allparts.append(p)
pd.concat(allparts).to_csv(_SAL+"/S_t_consolidado.csv", index=False)
print(summary.to_string(index=False))
print(f"\n{len(rows)} sujetos procesados.")
