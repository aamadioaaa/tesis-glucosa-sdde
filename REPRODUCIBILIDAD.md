# Notas de cómputo

Este documento reúne el origen del código de cada capítulo, los parámetros de cómputo y las
convenciones numéricas con las que se obtienen las cifras de la tesis. Cada notebook termina
con una tabla de verificación que compara automáticamente las cifras calculadas con las del
texto (`resultados/capN/verificacion_capN.csv`). El índice
[`INDICE_FIGURAS_Y_TABLAS.md`](INDICE_FIGURAS_Y_TABLAS.md) indica qué archivo regenera cada
figura y cada tabla.

## 1. Origen del código por capítulo

| Capítulo | Código |
|---|---|
| 1–2 | Capítulos teóricos, sin cómputo |
| 3 | Notebook del capítulo, reorganizado en el paquete `glucosa` |
| 4 | Código de calibración (`Simulador_DBT2.ipynb` / `Simulador_NDBT2.ipynb`), en `glucosa.sdde`. Las 40 trayectorias se regeneran bit a bit a partir de los parámetros archivados |
| 5 | Notebook calculado sobre las señales reales y simuladas, y script original de las Figuras 5.1–5.3 (`originales/cap5_figuras`) |
| 6 | Ecuaciones (6.1)–(6.10) implementadas en `glucosa.intervencion` |
| 7 | Scripts originales del detector, del alba y del ajuste del modelo (`originales/cap7_cgmacros`); control de calidad, plano H × C e intervención en el paquete `glucosa` |

## 2. Datos

1. **DBT2-04 y DBT2-06 son la misma serie** (los archivos `subject_4_E.csv` y
   `subject_6_E.csv` contienen los mismos 1200 valores). El grupo DBT2 tiene 19 registros
   distintos; una prueba automática (`tests/test_reproduccion.py`) lo documenta.
2. **El Capítulo 3 usa 19 registros DBT2** (excluye DBT2-11); los Capítulos 4–6 usan los 20.
   El submuestreo de 11 628 = C(19,5) combinaciones incluye las dos copias del mismo registro.
3. **Aumento de datos NDBT2 (DCT + PCA + GMM).** Las 15 series aumentadas se publican como
   datos.
4. **CGMacros.** Cada valor de 15 min es el promedio de las lecturas de un minuto del intervalo
   de reloj correspondiente (como en `extract_St.py`).

## 3. Parámetros de cómputo (Anexo B)

| Elemento | Valor | Dónde |
|---|---|---|
| S(t) del modelo, ec. (3.13) | ventana de 10 muestras, 35 % de los coeficientes DCT, raíz de la envolvente normalizada | `forzamiento.forzamiento` |
| S(t) para el exponente espectral (Sec. 3.3.5) | ventana de 17 muestras, 10 % de los coeficientes, sin raíz | `forzamiento.forzamiento_espectral_cap3` |
| S(t) del detector del Capítulo 7 | ventana de 10 muestras, 10 % de los coeficientes, sin raíz | `originales/cap7_cgmacros/extract_St.py` |
| Detector (Sec. 3.3.6) | Morlet complejo `cmor1.0-1.0` (Cap. 7: `cmor1.5-1.0`), 24 períodos por banda, W normalizada por su máximo, constantes de (3.35) nulas, ventanas cerradas | `detector.py` |
| Gradación del alba | fuerte: I_G, I_S > 0,5 y W̄ ≥ mediana del sujeto; posible: min(I_G, I_S) < 0,15; moderada en el resto | `detector.detectar_alba` |
| Falsos vecinos | criterio 1 de Kennel (cociente > 15), umbral del 5 % | `no_lineal.falsos_vecinos` |
| Retardo del integrador | ⌊τ_d/Δt⌋ pasos de 0,5 min | `sdde.simular` |
| Filtro del sensor | Savitzky–Golay de orden 3, `mode="nearest"`; las series NDBT2 se calibraron sin filtro | `sdde.simular` |
| Semilla | 42 para la realización del ruido (`numpy.random.seed(42)`) | `sdde.ruido_blanco` |
| Función objetivo (4.23) | pesos del texto; el término 500·ΔLLE está previsto pero no activo en la calibración informada (`sdde.ajustar(..., incluir_lle=True)` lo agrega); Δσ = \|σ_sim − σ_real\|/σ_real | `sdde.costo` |
| Descomposición de la Fig. 4.2 | incrementos de cada término acumulados en 15 min | notebook 04 §5 |
| Nivel de referencia con intervención (6.6b) | B_int(I_{k+1}) = B(I_{k+1}) + ⟨G_int⟩_{I_k} − ⟨G_ref⟩_{I_k} | `intervencion.simular_intervencion` |
| Función de costo (6.10) | w_A = 10, w_TBR = 100, w_e = 1, umbral de TBR 2 % | `intervencion.PESOS_COSTO` |
| Primer ensayo TPE | ρ_i = 1 (sin intervención) | `intervencion.optimizar` |
| Objetivo de HbA1c eq. | 6,8 % caso individual; 5,5 % lectura poblacional y CGMacros | notebooks 06–07 |
| Eventos del operador | detector de la Sec. 3.3.6.3 sobre la trayectoria de referencia | notebooks 06–07 |
| Derivada en A(t), ec. (6.7e) | diferencias hacia atrás en la grilla de 0,5 min | `intervencion._integrar_seguridad` |
| Reconstrucción DCT-PCA | ruido residual de 0,35 × el desvío residual | `cgmacros.BaseDCTPCA.fraccion_ruido` |
| Exclusiones en CGMacros | 007 (menos de 6 días utilizables) y 018 (más de 24 h reconstruidas, sólo en la validación de eventos) | notebook 07 |
| Parámetros de la intervención (Sec. 7.8) | recalibración con la función objetivo del Capítulo 4: 300 ensayos TPE, mejor de tres corridas (semillas 42, 1 y 2) | notebook 07 §6.1 |

## 4. Convenciones numéricas

* **Desviaciones estándar.** Las medias ± DE de la Tabla 4.2 y de la Sec. 4.5 usan ddof = 0;
  σ_E, CV_E, CV_S, L1 y Δσ del Capítulo 5 usan ddof = 1 (`error.metricas_error`).
* **Tablas 5.1–5.4.** β_n se redondea a 4 decimales y luego a 3; el MAE diario se redondea a
  0,1 y luego a entero («mitad hacia arriba»); Δ es el máximo menos el mínimo de los valores
  diarios redondeados a 0,1.
* **MAE/σ_E del NDBT2 (Sec. 5.9).** El valor 0,574 ± 0,051 se calcula sobre las 20 series; el
  contraste de Mann–Whitney usa los 5 registros reales.
* **Redondeo.** Las cifras se informan con redondeo «mitad hacia arriba».

## 5. Material original

* **Capítulo 7** (`originales/cap7_cgmacros`): scripts del detector, del alba, del plano H×C y
  del ajuste del modelo, con sus salidas. El ajuste usó `sim_langevin.py` con 120 ensayos TPE
  y ruido sembrado con `hash(sujeto)`, que Python aleatoriza en cada sesión; por eso las cifras
  de la Sec. 7.7 se calculan a partir del resumen y las trayectorias archivados
  (`archivado/`). Ver [`originales/cap7_cgmacros/LEEME.md`](originales/cap7_cgmacros/LEEME.md).
* **Capítulo 5** (`originales/cap5_figuras`): script de las Figuras 5.1–5.3.

## 6. Versiones de las bibliotecas

`requirements.txt` fija PyWavelets < 1.8: desde la versión 1.8 la wavelet `cmor` cambió
ligeramente y los valores de W se modifican en la tercera cifra.
