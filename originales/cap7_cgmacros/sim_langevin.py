"""
LANGEVIN GLUCOSE MODEL OPTIMIZER v5
====================================
Mejoras respecto a v4:
1. Separación clara: RESTRICCIONES DURAS → OBJETIVO → VALIDACIÓN
2. Pesos configurables en CONFIG
3. Diagnóstico de qué componente domina el score
4. Sin variables globales
5. Opción de deshabilitar penalties para experimentos

Autor: Optimizado para tesis doctoral
"""

import numpy as np
import pandas as pd
import glob
import os
import math
import json
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
from scipy.fftpack import dct, idct
from scipy.signal import savgol_filter
from numba import jit
import optuna
from tqdm import tqdm

# Intentar importar nolds, si no está disponible, usar fallback
try:
    import nolds
    HAS_NOLDS = True
except ImportError:
    HAS_NOLDS = False
    print("⚠️ nolds no instalado. Lyapunov deshabilitado.")

# =============================================================================
# CONFIGURACIÓN CENTRALIZADA
# =============================================================================
@dataclass
class Config:
    """Configuración completa del optimizador."""
    
    # --- RUTAS ---
    input_path: str = r"C:\Users\aru_a\OneDrive\1.Doctorado\8.Modelo_Langevin\datos\subject_*_E.csv"
    output_root: str = r"C:\Users\aru_a\OneDrive\1.Doctorado\8.Modelo_Langevin\datos\Resultados_DBT2_v5"
    
    # --- PARÁMETROS FÍSICOS ---
    points_per_day: int = 96
    dt_data: float = 15.0        # Minutos entre muestras
    dt_sim: float = 0.5          # Paso de simulación (minutos)
    limit_days: int = 12         # Días a procesar (None = todos)
    days_per_block: int = 3      # Días por bloque para basal dinámica
    
    # --- FILTRO SENSOR VIRTUAL ---
    avg_window_min: float = 15.0
    
    # --- LÍMITES FISIOLÓGICOS (RESTRICCIONES DURAS) ---
    g_min_survival: float = 40.0    # Por debajo = muerte
    g_max_survival: float = 500.0   # Por encima = coma
    g_min_safe: float = 54.0        # Hipoglucemia clínica
    g_max_safe: float = 250.0       # Hiperglucemia severa
    tir_min_viable: float = 0.10    # Mínimo 10% en rango para ser viable
    
    # --- BOUNDS DE PARÁMETROS ---
    alpha_bounds: Tuple[float, float] = (0.001, 0.05)
    tau_bounds: Tuple[float, float] = (5.0, 120.0)
    kf_bounds: Tuple[float, float] = (0.1, 10.0)
    beta_bounds: Tuple[float, float] = (0.05, 2.0)
    shift_bounds: Tuple[float, float] = (-20.0, 60.0)
    
    # --- PESOS DEL OBJETIVO (ajustar aquí) ---
    # Objetivo principal (ajuste de señal)
    w_rmse: float = 100.0
    w_corr: float = 150.0
    w_std: float = 50.0
    w_bias: float = 30.0
    
    # Penalties de propiedades (pueden deshabilitarse)
    w_tir: float = 400.0      # Tiempo en rango
    w_tbr: float = 800.0      # Tiempo bajo rango (más crítico)
    w_tar: float = 200.0      # Tiempo alto rango
    w_hxc: float = 300.0      # Entropía-Complejidad
    w_spectral: float = 150.0 # Exponente espectral β
    w_lyapunov: float = 0.0   # ← DESHABILITADO POR DEFECTO (usar como validación)
    
    # --- SWITCHES PARA EXPERIMENTOS ---
    use_clinical_penalty: bool = True
    use_hxc_penalty: bool = True
    use_spectral_penalty: bool = True
    use_lyapunov_penalty: bool = False  # ← OFF: usar solo como validación
    
    # --- OPTIMIZACIÓN ---
    n_trials: int = 300
    n_startup_trials: int = 30   # Trials aleatorios antes de TPE
    optuna_seed: int = 42
    noise_seed: int = None       # None = diferente para cada sujeto
    
    @property
    def ratio(self) -> int:
        return int(self.dt_data / self.dt_sim)
    
    @property
    def dt_sqrt(self) -> float:
        return np.sqrt(self.dt_sim)


# Instancia global de configuración (modificable)
CFG = Config()


# =============================================================================
# 1. HERRAMIENTAS MATEMÁTICAS
# =============================================================================
def analyze_noise_color(signal: np.ndarray, dt: float) -> Dict:
    """Calcula el exponente espectral β (PSD ∝ f^(-β))."""
    signal = np.asarray(signal)
    n = len(signal)
    fft_vals = np.fft.rfft(signal - np.mean(signal))
    fft_freq = np.fft.rfftfreq(n, d=dt)
    psd = np.abs(fft_vals) ** 2
    
    # Filtrar frecuencias válidas
    keep = (fft_freq > 1e-6) & (fft_freq < 0.4/dt)
    if np.sum(keep) < 10:
        return {'beta': 1.0}
    
    freqs_log = np.log10(fft_freq[keep])
    psd_log = np.log10(psd[keep] + 1e-10)
    
    # Ajuste lineal robusto
    slope, _ = np.polyfit(freqs_log, psd_log, 1)
    return {'beta': -slope}


def statistical_complexity(P: np.ndarray) -> Tuple[float, float]:
    """Calcula entropía normalizada H y complejidad C."""
    N = len(P)
    Pe = np.ones(N) / N
    nz = P[P > 0]
    
    if len(nz) == 0:
        return 0.0, 0.0
    
    H = -np.sum(nz * np.log(nz))
    H_norm = H / np.log(N)
    
    P_mix = 0.5 * (P + Pe)
    
    def entropy(p):
        p_nz = p[p > 0]
        return -np.sum(p_nz * np.log(p_nz))
    
    JSD = entropy(P_mix) - 0.5 * entropy(P) - 0.5 * entropy(Pe)
    
    k = (N + 1) / N
    Q0 = -2.0 / (k * np.log(N + 1) - 2 * np.log(2 * N) + np.log(N))
    C = H_norm * (JSD * Q0)
    
    return H_norm, C


def calculate_hxc(x: np.ndarray, m: int = 3, delay: int = 1) -> Tuple[float, float]:
    """Calcula H×C desde serie temporal usando patrones ordinales."""
    x = np.asarray(x)
    n = len(x)
    
    if n < m * delay:
        return 0.5, 0.0
    
    idx = np.arange(n - (m - 1) * delay)
    vectors = np.vstack([x[idx + i * delay] for i in range(m)]).T
    orders = np.argsort(vectors, axis=1)
    _, counts = np.unique(orders, axis=0, return_counts=True)
    
    P = counts / counts.sum()
    N_fact = math.factorial(m)
    P_full = np.zeros(N_fact)
    P_full[:len(P)] = P
    
    return statistical_complexity(P_full)


def calculate_clinical_metrics(signal: np.ndarray) -> Dict:
    """Calcula métricas clínicas de glucosa."""
    total = len(signal)
    if total == 0:
        return {'tbr': 0, 'tir': 0, 'tar': 0, 'tbr_severe': 0}
    
    return {
        'tbr_severe': np.sum(signal < 54) / total,   # Hipoglucemia severa
        'tbr': np.sum(signal < 70) / total,          # Hipoglucemia
        'tir': np.sum((signal >= 70) & (signal <= 180)) / total,  # En rango
        'tar': np.sum(signal > 180) / total,         # Hiperglucemia
        'tar_severe': np.sum(signal > 250) / total   # Hiperglucemia severa
    }


def get_lyapunov(signal: np.ndarray, dt_minutes: float = 15.0) -> float:
    """Calcula el Exponente de Lyapunov Más Grande (LLE)."""
    if not HAS_NOLDS:
        return np.nan
    
    try:
        lle = nolds.lyap_r(
            signal,
            emb_dim=3,
            lag=1,
            min_tsep=10,
            trajectory_len=20,
            fit="poly",
            debug_plot=False
        )
        # Convertir a 1/hora
        return lle * (60.0 / dt_minutes)
    except:
        return np.nan


# =============================================================================
# 2. GENERADOR DE FORZAMIENTO S(t)
# =============================================================================
class ForcingGenerator:
    """Genera el término de forzamiento S(t) a partir de la glucosa."""
    
    def __init__(self, cutoff_fraction: float = 0.35):
        self.cutoff_fraction = cutoff_fraction
        self.E_norm = None
        self.grad = None
    
    def compute_S(self, glucose: np.ndarray) -> np.ndarray:
        """
        Calcula S(t) = tanh(grad/σ) × σ × (0.2 + 0.8√E)
        
        Donde:
        - grad = dG/dt (gradiente)
        - E = energía local normalizada (varianza rolling filtrada por DCT)
        """
        n = len(glucose)
        
        # 1. Energía local (varianza rolling)
        energy = pd.Series(glucose).rolling(window=10, center=True).var().fillna(0).values
        
        # 2. Filtrado DCT (suavizado)
        coeffs = dct(energy, type=2, norm='ortho')
        coeffs[int(n * self.cutoff_fraction):] = 0
        E_smooth = np.maximum(0, idct(coeffs, type=2, norm='ortho'))
        
        # 3. Normalizar
        E_max = np.max(E_smooth)
        self.E_norm = E_smooth / E_max if E_max > 0 else E_smooth
        
        # 4. Gradiente
        self.grad = np.gradient(glucose, CFG.dt_data)
        grad_std = np.std(self.grad) + 1e-6
        
        # 5. Forzamiento
        S = np.tanh(self.grad / (grad_std * 0.5)) * grad_std * (0.2 + 0.8 * np.sqrt(self.E_norm))
        
        return S


# =============================================================================
# 3. MOTOR DE SIMULACIÓN (Numba JIT)
# =============================================================================
@jit(nopython=True, cache=True)
def simulate_langevin_core(
    S_sim: np.ndarray,
    alpha: float,
    tau_steps: int,
    k_f: float,
    beta_n: float,
    basal_vec: np.ndarray,
    basal_shift: float,
    G0: float,
    G_min: float,
    G_max: float,
    noise: np.ndarray,
    dt_sim: float,
    dt_sqrt: float
) -> np.ndarray:
    """
    Integración numérica de Euler-Maruyama para:
    dG = [-α(G(t-τ) - B - Δb) + k_f·S(t)]dt + β·dW
    """
    n_steps = len(S_sim)
    G = np.zeros(n_steps, dtype=np.float64)
    G[:tau_steps + 1] = G0
    
    for t in range(tau_steps, n_steps - 1):
        # Basal target
        target = basal_vec[t] + basal_shift
        
        # Drift determinista
        drift = -alpha * (G[t - tau_steps] - target) + k_f * S_sim[t]
        
        # Limitar drift para estabilidad numérica
        drift = max(-60.0, min(60.0, drift))
        
        # Euler-Maruyama
        G[t + 1] = G[t] + drift * dt_sim + beta_n * noise[t] * dt_sqrt
        
        # Clipping fisiológico
        G[t + 1] = max(G_min, min(G_max, G[t + 1]))
    
    return G


def simulate_langevin(params: Dict, precalc: Dict) -> np.ndarray:
    """Wrapper de alto nivel para simulación."""
    
    tau_steps = int(params['tau'] / CFG.dt_sim)
    
    # Simulación fina
    G_full = simulate_langevin_core(
        S_sim=precalc['S_sim'],
        alpha=params['alpha'],
        tau_steps=tau_steps,
        k_f=params['k_f'],
        beta_n=params['beta'],
        basal_vec=precalc['basal_vec'],
        basal_shift=params['shift'],
        G0=precalc['G0'],
        G_min=CFG.g_min_survival,
        G_max=CFG.g_max_survival,
        noise=precalc['noise'],
        dt_sim=CFG.dt_sim,
        dt_sqrt=CFG.dt_sqrt
    )
    
    # Filtro Savitzky-Golay (sensor virtual)
    window_len = int(CFG.avg_window_min / CFG.dt_sim)
    if window_len % 2 == 0:
        window_len += 1
    
    if window_len >= 5:
        try:
            G_smooth = savgol_filter(G_full, window_length=window_len, polyorder=3, mode='nearest')
            G_smooth[0] = G_full[0]
        except:
            G_smooth = G_full
    else:
        G_smooth = G_full
    
    # Submuestreo a frecuencia de datos
    return G_smooth[::CFG.ratio][:precalc['n_data']]


# =============================================================================
# 4. FUNCIÓN OBJETIVO CON ARQUITECTURA CLARA
# =============================================================================
@dataclass
class ScoreBreakdown:
    """Desglose del score para diagnóstico."""
    total: float = 0.0
    rmse: float = 0.0
    corr: float = 0.0
    std_err: float = 0.0
    bias: float = 0.0
    clinical: float = 0.0
    hxc: float = 0.0
    spectral: float = 0.0
    lyapunov: float = 0.0
    rejected: bool = False
    reject_reason: str = ""


def evaluate_solution(
    params: Dict,
    g_real: np.ndarray,
    precalc: Dict,
    targets: Dict,
    return_breakdown: bool = False
) -> float:
    """
    Evalúa una solución con arquitectura de 3 capas:
    
    CAPA 1: RESTRICCIONES DURAS (rechazo inmediato)
    CAPA 2: OBJETIVO (ajuste de señal)
    CAPA 3: PENALTIES OPCIONALES (propiedades emergentes)
    """
    breakdown = ScoreBreakdown()
    
    # =========================================================================
    # SIMULACIÓN
    # =========================================================================
    try:
        G_sim = simulate_langevin(params, precalc)
    except Exception as e:
        breakdown.rejected = True
        breakdown.reject_reason = f"Simulación falló: {e}"
        breakdown.total = 1e10
        return (breakdown.total, breakdown) if return_breakdown else breakdown.total
    
    # =========================================================================
    # CAPA 1: RESTRICCIONES DURAS (Rechazo)
    # =========================================================================
    
    # 1.1 Hipoglucemia severa (muerte)
    mask_hypo = G_sim < CFG.g_min_survival
    if np.any(mask_hypo):
        depth = np.sum(CFG.g_min_survival - G_sim[mask_hypo])
        breakdown.rejected = True
        breakdown.reject_reason = f"Hipoglucemia mortal ({np.sum(mask_hypo)} puntos)"
        breakdown.total = 1e6 + depth * 1000
        return (breakdown.total, breakdown) if return_breakdown else breakdown.total
    
    # 1.2 Señal degenerada
    if np.std(G_sim) < 2.0:
        breakdown.rejected = True
        breakdown.reject_reason = "Señal constante (std < 2)"
        breakdown.total = 1e6
        return (breakdown.total, breakdown) if return_breakdown else breakdown.total
    
    # 1.3 TIR mínimo viable
    clin_sim = calculate_clinical_metrics(G_sim)
    if clin_sim['tir'] < CFG.tir_min_viable:
        breakdown.rejected = True
        breakdown.reject_reason = f"TIR demasiado bajo ({clin_sim['tir']*100:.1f}%)"
        breakdown.total = 1e5 * (1 - clin_sim['tir'])
        return (breakdown.total, breakdown) if return_breakdown else breakdown.total
    
    # =========================================================================
    # CAPA 2: OBJETIVO PRINCIPAL (Ajuste de señal)
    # =========================================================================
    
    # RMSE
    rmse = np.sqrt(np.mean((G_sim - g_real) ** 2))
    breakdown.rmse = rmse * CFG.w_rmse
    
    # Correlación
    corr = np.corrcoef(G_sim, g_real)[0, 1]
    if np.isnan(corr):
        corr = 0.0
    breakdown.corr = (1 - corr) * CFG.w_corr
    
    # Error de amplitud (desviación estándar)
    std_real = np.std(g_real)
    std_err = abs(np.std(G_sim) - std_real) / (std_real + 1e-6)
    breakdown.std_err = std_err * CFG.w_std
    
    # Bias (sesgo sistemático)
    bias = abs(np.mean(G_sim) - np.mean(g_real))
    breakdown.bias = bias * CFG.w_bias
    
    # =========================================================================
    # CAPA 3: PENALTIES OPCIONALES (Propiedades emergentes)
    # =========================================================================
    
    # 3.1 Clinical (TIR/TBR/TAR)
    if CFG.use_clinical_penalty:
        clin_real = targets['clinical']
        err_tbr = abs(clin_real['tbr'] - clin_sim['tbr'])
        err_tir = abs(clin_real['tir'] - clin_sim['tir'])
        err_tar = abs(clin_real['tar'] - clin_sim['tar'])
        breakdown.clinical = err_tbr * CFG.w_tbr + err_tir * CFG.w_tir + err_tar * CFG.w_tar
    
    # 3.2 H×C (Entropía-Complejidad)
    if CFG.use_hxc_penalty:
        try:
            H_sim, C_sim = calculate_hxc(G_sim)
            dist_hxc = np.sqrt((H_sim - targets['H']) ** 2 + (C_sim - targets['C']) ** 2)
            breakdown.hxc = dist_hxc * CFG.w_hxc
        except:
            breakdown.hxc = CFG.w_hxc * 0.5  # Penalty moderado si falla
    
    # 3.3 Espectral (color del ruido)
    if CFG.use_spectral_penalty:
        try:
            spec_sim = analyze_noise_color(G_sim, CFG.dt_data)
            diff_beta = abs(targets['beta_spectral'] - spec_sim['beta'])
            breakdown.spectral = diff_beta * CFG.w_spectral
        except:
            breakdown.spectral = CFG.w_spectral * 0.5
    
    # 3.4 Lyapunov (estabilidad dinámica)
    if CFG.use_lyapunov_penalty and CFG.w_lyapunov > 0:
        lle_sim = get_lyapunov(G_sim, CFG.dt_data)
        if not np.isnan(lle_sim) and not np.isnan(targets.get('lle', np.nan)):
            diff_lle = abs(targets['lle'] - lle_sim)
            breakdown.lyapunov = diff_lle * CFG.w_lyapunov
    
    # =========================================================================
    # SCORE TOTAL
    # =========================================================================
    breakdown.total = (
        breakdown.rmse +
        breakdown.corr +
        breakdown.std_err +
        breakdown.bias +
        breakdown.clinical +
        breakdown.hxc +
        breakdown.spectral +
        breakdown.lyapunov
    )
    
    return (breakdown.total, breakdown) if return_breakdown else breakdown.total


# =============================================================================
# 5. OPTIMIZACIÓN CON OPTUNA
# =============================================================================
class LangevinOptimizer:
    """Optimizador del modelo de Langevin."""
    
    def __init__(self, g_real: np.ndarray, subject_id: str):
        self.g_real = g_real
        self.subject_id = subject_id
        self.precalc = None
        self.targets = None
        self.generator = None
        self.best_breakdown = None
        
    def prepare(self, noise_seed: Optional[int] = None):
        """Prepara datos precalculados y targets."""
        n_data = len(self.g_real)
        n_sim = n_data * CFG.ratio
        
        # Generador de forzamiento
        self.generator = ForcingGenerator()
        S_real = self.generator.compute_S(self.g_real)
        S_sim = np.repeat(S_real, CFG.ratio)[:n_sim].astype(np.float64)
        
        # Basal dinámica
        basal_vec = self._get_dynamic_basal(n_sim)
        
        # Ruido
        if noise_seed is not None:
            np.random.seed(noise_seed)
        else:
            np.random.seed(hash(self.subject_id) % (2**32))
        noise = np.random.randn(n_sim).astype(np.float64)
        
        self.precalc = {
            'S_sim': S_sim,
            'S_real': S_real,
            'basal_vec': basal_vec,
            'noise': noise,
            'G0': self.g_real[0],
            'n_data': n_data
        }
        
        # Targets
        H_real, C_real = calculate_hxc(self.g_real)
        spec_real = analyze_noise_color(self.g_real, CFG.dt_data)
        clin_real = calculate_clinical_metrics(self.g_real)
        lle_real = get_lyapunov(self.g_real, CFG.dt_data)
        
        self.targets = {
            'H': H_real,
            'C': C_real,
            'beta_spectral': spec_real['beta'],
            'clinical': clin_real,
            'lle': lle_real
        }
        
    def _get_dynamic_basal(self, n_sim: int) -> np.ndarray:
        """Calcula basal dinámica por bloques."""
        basal_vec = np.zeros(n_sim, dtype=np.float64)
        block_size_real = CFG.points_per_day * CFG.days_per_block
        block_size_sim = block_size_real * CFG.ratio
        n_blocks = int(np.ceil(len(self.g_real) / block_size_real)) + 1
        
        for i in range(n_blocks):
            start_real = i * block_size_real
            end_real = min((i + 1) * block_size_real, len(self.g_real))
            if start_real >= len(self.g_real):
                break
            block_mean = np.mean(self.g_real[start_real:end_real])
            start_sim = i * block_size_sim
            end_sim = min((i + 1) * block_size_sim, n_sim)
            basal_vec[start_sim:end_sim] = block_mean
            
        return basal_vec
    
    def _objective(self, trial: optuna.Trial) -> float:
        """Función objetivo para Optuna."""
        params = {
            'alpha': trial.suggest_float("alpha", *CFG.alpha_bounds),
            'tau': trial.suggest_float("tau", *CFG.tau_bounds),
            'k_f': trial.suggest_float("k_f", *CFG.kf_bounds),
            'beta': trial.suggest_float("beta", *CFG.beta_bounds),
            'shift': trial.suggest_float("shift", *CFG.shift_bounds)
        }
        
        score, breakdown = evaluate_solution(
            params, self.g_real, self.precalc, self.targets,
            return_breakdown=True
        )
        
        # Pruning para soluciones rechazadas
        if breakdown.rejected:
            raise optuna.TrialPruned()
        
        return score
    
    def optimize(self, verbose: bool = True) -> Dict:
        """Ejecuta la optimización."""
        if self.precalc is None:
            self.prepare()
        
        # Configurar Optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING if not verbose else optuna.logging.INFO)
        
        sampler = optuna.samplers.TPESampler(
            seed=CFG.optuna_seed,
            n_startup_trials=CFG.n_startup_trials
        )
        
        study = optuna.create_study(direction="minimize", sampler=sampler)
        
        # Callbacks para diagnóstico
        def callback(study, trial):
            if trial.state == optuna.trial.TrialState.COMPLETE:
                if verbose and trial.number % 50 == 0:
                    print(f"  Trial {trial.number}: score={trial.value:.2f}")
        
        study.optimize(
            self._objective,
            n_trials=CFG.n_trials,
            callbacks=[callback],
            catch=(Exception,)
        )
        
        # Mejor solución
        best_params = study.best_params
        
        # Evaluar con breakdown completo
        _, self.best_breakdown = evaluate_solution(
            best_params, self.g_real, self.precalc, self.targets,
            return_breakdown=True
        )
        
        return best_params
    
    def get_diagnostics(self, params: Dict) -> Dict:
        """Genera diagnósticos completos de una solución."""
        G_sim = simulate_langevin(params, self.precalc)
        
        # Métricas de ajuste
        corr = np.corrcoef(G_sim, self.g_real)[0, 1]
        rmse = np.sqrt(np.mean((G_sim - self.g_real) ** 2))
        bias = np.mean(G_sim) - np.mean(self.g_real)
        amp_err = abs(np.std(G_sim) - np.std(self.g_real)) / np.std(self.g_real)
        
        # H×C
        H_sim, C_sim = calculate_hxc(G_sim)
        
        # Espectral
        spec_sim = analyze_noise_color(G_sim, CFG.dt_data)
        
        # Lyapunov
        lle_sim = get_lyapunov(G_sim, CFG.dt_data)
        
        # Clinical
        clin_sim = calculate_clinical_metrics(G_sim)
        
        return {
            'G_sim': G_sim,
            'metrics': {
                'corr': corr,
                'rmse': rmse,
                'bias': bias,
                'amp_err': amp_err
            },
            'hxc': {'H_sim': H_sim, 'C_sim': C_sim, 'H_real': self.targets['H'], 'C_real': self.targets['C']},
            'spectral': {'beta_sim': spec_sim['beta'], 'beta_real': self.targets['beta_spectral']},
            'lyapunov': {'lle_sim': lle_sim, 'lle_real': self.targets['lle']},
            'clinical': {'sim': clin_sim, 'real': self.targets['clinical']},
            'breakdown': self.best_breakdown
        }


# =============================================================================
# 6. GUARDADO DE RESULTADOS
# =============================================================================
def save_results(
    subject_id: str,
    g_real: np.ndarray,
    params: Dict,
    diagnostics: Dict,
    generator: ForcingGenerator,
    output_root: str
):
    """Guarda todos los resultados de un sujeto."""
    # Carpeta: sim_subject_1, sim_subject_2, etc.
    folder = os.path.join(output_root, f"sim_subject_{subject_id}")
    os.makedirs(folder, exist_ok=True)
    
    G_sim = diagnostics['G_sim']
    
    # 1. Parámetros y métricas
    params_dict = {
        "ID": subject_id,
        "Fitted_Parameters": {
            "alpha": params['alpha'],
            "tau": params['tau'],
            "k_f": params['k_f'],
            "beta": params['beta'],
            "basal_shift": params['shift']
        },
        "Metrics": {
            "Correlation": diagnostics['metrics']['corr'],
            "RMSE": diagnostics['metrics']['rmse'],
            "Bias": diagnostics['metrics']['bias'],
            "Amplitude_Error": diagnostics['metrics']['amp_err'],
            "Real_Glucose_Mean": float(np.mean(g_real)),
            "Real_Glucose_Std": float(np.std(g_real))
        },
        "HxC": diagnostics['hxc'],
        "Spectral": diagnostics['spectral'],
        "Lyapunov": diagnostics['lyapunov'],
        "Clinical": {
            "Real": diagnostics['clinical']['real'],
            "Sim": diagnostics['clinical']['sim']
        },
        "Score_Breakdown": {
            "total": diagnostics['breakdown'].total,
            "rmse": diagnostics['breakdown'].rmse,
            "corr": diagnostics['breakdown'].corr,
            "std_err": diagnostics['breakdown'].std_err,
            "bias": diagnostics['breakdown'].bias,
            "clinical": diagnostics['breakdown'].clinical,
            "hxc": diagnostics['breakdown'].hxc,
            "spectral": diagnostics['breakdown'].spectral,
            "lyapunov": diagnostics['breakdown'].lyapunov
        }
    }
    
    # Archivo: param_subject_1.json
    with open(os.path.join(folder, f"param_subject_{subject_id}.json"), 'w') as f:
        json.dump(params_dict, f, indent=4, default=float)
    
    # 2. Metabolismo (S, E, grad)
    metabolism_dict = {
        "Description": "Componentes del forzamiento S(t)",
        "Vectors": {
            "S_forcing": generator.grad.tolist(),  # S se deriva del gradiente
            "Energy_Normalized": generator.E_norm.tolist(),
            "Gradient": generator.grad.tolist()
        }
    }
    
    # Archivo: metabolism_subject_1.json
    with open(os.path.join(folder, f"metabolism_subject_{subject_id}.json"), 'w') as f:
        json.dump(metabolism_dict, f, indent=4)
    
    # 3. Glucosa simulada
    glucose_dict = {
        "ID": subject_id,
        "Sampling_Minutes": CFG.dt_data,
        "Glucose_Simulated": G_sim.tolist()
    }
    
    # Archivo: glucose_sim_subject_1.json
    with open(os.path.join(folder, f"glucose_sim_subject_{subject_id}.json"), 'w') as f:
        json.dump(glucose_dict, f, indent=4)
    
    print(f"  💾 Guardado en {folder}")


# =============================================================================
# 7. PROCESO PRINCIPAL
# =============================================================================
def process_subject(filepath: str, verbose: bool = True) -> Optional[Dict]:
    """Procesa un sujeto completo."""
    
    # Extraer ID (solo el número, sin _E ni _S)
    # subject_1_E.csv → 1
    # subject_12_S.csv → 12
    filename = os.path.basename(filepath).split('.')[0]  # subject_1_E
    parts = filename.replace('subject_', '').replace('_E', '').replace('_S', '')
    subject_id = parts  # Solo el número: "1", "12", etc.
    
    # Cargar datos
    try:
        df = pd.read_csv(filepath)
        if 'Glucosa' in df.columns:
            g_real = df['Glucosa'].values.astype(float)
        else:
            g_real = df.iloc[:, 0].values.astype(float)
    except Exception as e:
        print(f"❌ Error cargando {subject_id}: {e}")
        return None
    
    # Limitar días
    if CFG.limit_days:
        g_real = g_real[:CFG.limit_days * CFG.points_per_day]
    
    if verbose:
        print(f"\n📂 {subject_id}")
        print(f"   Puntos: {len(g_real)}, G₀: {g_real[0]:.0f}, Ḡ: {np.mean(g_real):.1f} mg/dL")
    
    # Optimizar
    optimizer = LangevinOptimizer(g_real, subject_id)
    optimizer.prepare(noise_seed=CFG.noise_seed)
    
    best_params = optimizer.optimize(verbose=False)
    diagnostics = optimizer.get_diagnostics(best_params)
    
    # Mostrar diagnóstico
    if verbose:
        bd = diagnostics['breakdown']
        print(f"   Score: {bd.total:.1f} = RMSE({bd.rmse:.1f}) + Corr({bd.corr:.1f}) + "
              f"Clin({bd.clinical:.1f}) + HxC({bd.hxc:.1f}) + Spec({bd.spectral:.1f})")
        print(f"   α={best_params['alpha']:.6f}, τ={best_params['tau']:.1f}, "
              f"kf={best_params['k_f']:.2f}, β={best_params['beta']:.3f}")
        print(f"   Corr={diagnostics['metrics']['corr']:.3f}, "
              f"LLE_err={abs(diagnostics['lyapunov']['lle_sim']-diagnostics['lyapunov']['lle_real']):.3f}")
    
    # Guardar
    save_results(
        subject_id, g_real, best_params, diagnostics,
        optimizer.generator, CFG.output_root
    )
    
    return {
        'ID': subject_id,
        'params': best_params,
        'diagnostics': diagnostics
    }


def run_batch(input_pattern: Optional[str] = None, verbose: bool = True):
    """Procesa todos los sujetos."""
    
    pattern = input_pattern or CFG.input_path
    files = sorted(glob.glob(pattern))
    
    print("=" * 80)
    print(f"LANGEVIN OPTIMIZER v5")
    print(f"Sujetos: {len(files)}")
    print(f"Output: {CFG.output_root}")
    print(f"Penalties activos: Clinical={CFG.use_clinical_penalty}, "
          f"HxC={CFG.use_hxc_penalty}, Spectral={CFG.use_spectral_penalty}, "
          f"Lyapunov={CFG.use_lyapunov_penalty}")
    print("=" * 80)
    
    os.makedirs(CFG.output_root, exist_ok=True)
    
    results = []
    for filepath in tqdm(files, desc="Procesando", disable=verbose):
        try:
            result = process_subject(filepath, verbose=verbose)
            if result:
                results.append(result)
        except Exception as e:
            print(f"❌ Error en {filepath}: {e}")
    
    print(f"\n✅ Completado: {len(results)}/{len(files)} sujetos")
    return results


# =============================================================================
# EJECUCIÓN
# =============================================================================
