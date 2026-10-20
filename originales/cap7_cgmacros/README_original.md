# Análisis dinámico de CGMacros — paquete de reproducibilidad

Pipeline completo del Capítulo 7: extracción de S(t), plano H×C, detección de
ingestas y fenómeno del alba, y simulación del modelo de Langevin, sobre el
conjunto **CGMacros** (PhysioNet, 45 sujetos).

## Datos de salida (CSV)

| Archivo | Contenido |
|---|---|
| `CGMacros_datos_maestro.csv` | Una fila por sujeto: grupo clínico, demografía, H×C, ingestas (recall/precisión/niveles), alba, y parámetros/resultados de la simulación. |
| `series_G_S_45sujetos.csv` | Serie temporal (15 min) de los 45 sujetos: `G` (glucosa Libre), `v`, `v_cond`, `E_local`, `E_hat`, `S` (forzamiento). ~47 k filas. |
| `series_simulacion_diabeticos.csv` | Glucosa real vs. simulada por Langevin (18 diabéticos), por paso de 15 min. |
| `consolidado_3grupos.csv` | Detección + H×C por sujeto (base del maestro). |
| `agg_3grupos.csv` | Agregados por grupo (sano/prediabético/diabético). |
| `sim_CGM_diab_resumen.csv` | Métricas y parámetros ajustados de la simulación por sujeto. |

## Scripts (orden de ejecución)

1. `extract_St.py` — extrae S(t) de cada `CGMacros-XXX.csv` (ec. 3.6–3.13). Genera `S_t/`.
2. `plano_HC_libre_full.py` — plano H×C sobre Libre completo, 3 grupos + fronteras.
3. `detect_diabeticos.py` / `detect_sanos.py` — detección de ingestas (niveles) y alba (ΔG≥30, corte 08:00).
4. `consolidar_3grupos.py` — corre la detección uniforme en los 45 y agrega por grupo.
5. `sim_langevin.py` — módulo del modelo (extraído de `Simulador_DBT2.ipynb`, celda 2).
6. `run_sim_cgm.py` — corre la simulación sobre los diabéticos de CGMacros (`cgm_diab_in/`).
7. `sim_analisis_CGM.png`, `comparacion_3grupos.png`, `plano_HC_libre_full.png`,
   `grid_diabeticos.png` — figuras del capítulo.

## Parámetros clave

- Muestreo de trabajo: 15 min. Fuente: sensor **Libre** (más completo).
- S(t): corte DCT `cutoff_fraction = 0.35`, γ = 0.5, ventana E_local = 10 muestras.
- H×C: Bandt–Pompe D=3, τ=1; complejidad de Jensen–Shannon.
- Ingestas: ventana 2,5 h, ΔG≥20, banda wavelet rápida 0,5–2 h; niveles fuerte/moderada/leve.
- Alba: ventana 03:00–08:00 (inclusive), banda lenta 3–8 h, umbral **ΔG≥30 mg/dL**.
- Langevin: Euler–Maruyama dt=0,5 min; ajuste Optuna (120 trials) de α, τ, k_f, β, shift.

## Requisitos

`python3` con `pandas numpy scipy pywavelets numba optuna matplotlib`.
El conjunto CGMacros debe estar en la carpeta de datos (una subcarpeta `CGMacros-XXX` por sujeto).

## Cita del dataset

CGMacros — PhysioNet. Citar según la referencia oficial del repositorio.
