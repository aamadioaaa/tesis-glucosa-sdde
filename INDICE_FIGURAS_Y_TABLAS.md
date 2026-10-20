# Índice de figuras y tablas de la tesis

Cada figura y tabla de la tesis, con el archivo que la regenera y la sección del notebook que
la produce. Las cifras del texto se comparan en la tabla de verificación al final de cada
notebook (`resultados/capN/verificacion_capN.csv`).

## Figuras

| Figura | Contenido | Archivo | Notebook |
|---|---|---|---|
| 1.1 | Esquema conceptual: sistema circadiano, regulación hormonal y dinámica glucémica | — (esquema, sin cómputo) | — |
| 2.1 | Principio de medición del sensor CGM | — (esquema, sin cómputo) | — |
| 2.2 | Cuadrícula de error consenso del sensor (literatura) | — (figura de referencia, sin cómputo) | — |
| 3.1 | Registros de 24 h NDBT2 y DBT2 | `resultados/cap3/figura_3_1_registros_24h.png` | [03 §0](notebooks/03_analisis_de_los_datos.ipynb) |
| 3.2 | PSD media de la glucosa por grupo | `resultados/cap3/figura_3_2_psd_glucosa.png` | [03 §1](notebooks/03_analisis_de_los_datos.ipynb) |
| 3.3 | Exponente DFA por grupo | `resultados/cap3/figura_3_3_dfa.png` | [03 §2](notebooks/03_analisis_de_los_datos.ipynb) |
| 3.4–3.6 | Velocidad acondicionada, envolvente DCT y S(t) | `resultados/cap3/figuras_3_4_a_3_6_forzamiento.png` | [03 §3](notebooks/03_analisis_de_los_datos.ipynb) |
| 3.7 | Espectro de S(t) por grupo | `resultados/cap3/figura_3_7_psd_forzamiento.png` | [03 §3](notebooks/03_analisis_de_los_datos.ipynb) |
| 3.8 | Fenómeno del alba en DBT2-05 (12 días) | `resultados/cap3/figura_3_8_alba_DBT2_05.png` | [03 §4](notebooks/03_analisis_de_los_datos.ipynb) |
| 3.9 | Evento fuerte del alba aislado (DBT2-05, día 4) | `resultados/cap3/figura_3_9_alba_fuerte.png` | [03 §4.1](notebooks/03_analisis_de_los_datos.ipynb) |
| 3.10 | Clasificaciones del alba en otros sujetos | `resultados/cap3/figura_3_10_alba_clasificaciones.png` | [03 §4.1](notebooks/03_analisis_de_los_datos.ipynb) |
| 3.11 | Ventanas descartadas por hipoglucemia nocturna | `resultados/cap3/figura_3_11_exclusion_hipoglucemia.png` | [03 §4.1](notebooks/03_analisis_de_los_datos.ipynb) |
| 3.12 | Eventos compatibles con ingesta en DBT2-05 | `resultados/cap3/figura_3_12_ingestas_DBT2_05.png` | [03 §4](notebooks/03_analisis_de_los_datos.ipynb) |
| 3.13 | Clasificaciones de ingesta en otros sujetos | `resultados/cap3/figura_3_13_ingesta_clasificaciones.png` | [03 §4.1](notebooks/03_analisis_de_los_datos.ipynb) |
| 3.14 | Patrones ordinales para D = 3 | `resultados/cap3/figura_3_14_patrones_ordinales.png` | [03 §6](notebooks/03_analisis_de_los_datos.ipynb) |
| 3.15 | Plano H × C por sujeto | `resultados/cap3/figura_3_15_plano_HxC.png` | [03 §6](notebooks/03_analisis_de_los_datos.ipynb) |
| 3.16 | Selección de τ_emb y m_emb (AMI y FNN) | `resultados/cap3/figura_3_16_incrustacion.png` | [03 §8](notebooks/03_analisis_de_los_datos.ipynb) |
| 3.17 | Proyección de Takens | `resultados/cap3/figura_3_17_takens.png` | [03 §8](notebooks/03_analisis_de_los_datos.ipynb) |
| 4.1 | Panel de validación de DBT2-01 | `resultados/cap4/figura_4_1_panel_DBT2_01.png` | [04 §4](notebooks/04_el_modelo.ipynb) |
| 4.2 | Descomposición de la varianza por término | `resultados/cap4/figura_4_2_descomposicion.png` | [04 §5](notebooks/04_el_modelo.ipynb) |
| 5.1 | Análisis global del error, DBT2 | `resultados/cap5/figura_5_1_DBT2.png y original/figura_5_1_DBT2.{png,pdf}` | [05 §2 y §7](notebooks/05_analisis_del_error.ipynb) |
| 5.2 | Análisis global del error, NDBT2 | `resultados/cap5/figura_5_2_NDBT2.png y original/figura_5_2_NDBT2.{png,pdf}` | [05 §2 y §7](notebooks/05_analisis_del_error.ipynb) |
| 5.3 | Comparación DBT2–NDBT2 | `resultados/cap5/figura_5_3_comparacion.png y original/figura_5_3_comparativa.{png,pdf}` | [05 §3 y §7](notebooks/05_analisis_del_error.ipynb) |
| 5.4 | Compresión de amplitud y discrepancia de variabilidad | `resultados/cap5/figura_5_4_amplitud.png` | [05 §4](notebooks/05_analisis_del_error.ipynb) |
| 5.5 | Resolución diaria del error | `resultados/cap5/figura_5_5_resolucion_diaria.png` | [05 §5](notebooks/05_analisis_del_error.ipynb) |
| 5.6 | Error con resolución de 15 min | `resultados/cap5/figura_5_6_resolucion_15min.png` | [05 §5](notebooks/05_analisis_del_error.ipynb) |
| 5.7 | Grilla de error de Clarke | `resultados/cap5/figura_5_7_clarke.png` | [05 §6](notebooks/05_analisis_del_error.ipynb) |
| 6.1 | Descomposición starlet de S(t), sujeto 5 | `resultados/cap6/figura_6_1_starlet.png` | [06 §1](notebooks/06_optimizacion_control_glucemico.ipynb) |
| 6.2 | Acción local del operador | `resultados/cap6/figura_6_2_operador.png` | [06 §2](notebooks/06_optimizacion_control_glucemico.ipynb) |
| 6.3–6.4 | Energía por evento y trayectorias | `resultados/cap6/figura_6_3_energia_eventos.png`, `figura_6_4_trayectorias.png` | [06 §3](notebooks/06_optimizacion_control_glucemico.ipynb) |
| 6.5 | Atenuación uniforme | `resultados/cap6/figura_6_5_uniforme.png` | [06 §4](notebooks/06_optimizacion_control_glucemico.ipynb) |
| 6.6 | Tiempos en rango y HbA1c eq., antes y después | `resultados/cap6/figura_6_6_tiempos_en_rango_a1c.png` | [06 §3](notebooks/06_optimizacion_control_glucemico.ipynb) |
| 6.7–6.8 | OPTM y plano H × C del sujeto 5 | `resultados/cap6/figura_6_7_optm.png`, `figura_6_8_hxc.png` | [06 §4](notebooks/06_optimizacion_control_glucemico.ipynb) |
| 6.9 | Anatomía de los dos eventos de mayor reducción | `resultados/cap6/figura_6_9_anatomia_eventos.png` | [06 §3](notebooks/06_optimizacion_control_glucemico.ipynb) |
| 6.10 | Convergencia de la búsqueda TPE | `resultados/cap6/figura_6_10_convergencia.png` | [06 §3](notebooks/06_optimizacion_control_glucemico.ipynb) |
| 6.11 | PSD de la glucosa con y sin intervención | `resultados/cap6/figura_6_11_psd_intervencion.png` | [06 §6](notebooks/06_optimizacion_control_glucemico.ipynb) |
| 6.12 | Norma ‖v‖ en 20 sujetos | `resultados/cap6/figura_6_12_norma_v.png` | [06 §6](notebooks/06_optimizacion_control_glucemico.ipynb) |
| 7.1 | Reconstrucción del sujeto 018 y plano H × C | `resultados/cap7/figura_7_1_reconstruccion_y_hxc.png` | [07 §1–2](notebooks/07_cgmacros.ipynb) |
| 7.2 | Sujeto 029: comidas y detecciones | `resultados/cap7/figura_7_2_sujeto_029.png` | [07 §3.2](notebooks/07_cgmacros.ipynb) |
| 7.3 | Comparación de las tres condiciones | `resultados/cap7/figura_7_3_comparacion_grupos.png` | [07 §4](notebooks/07_cgmacros.ipynb) |
| 7.4 | Panel de validación de CGMacros-035 | `resultados/cap7/figura_7_4_panel_035.png` | [07 §5](notebooks/07_cgmacros.ipynb) |
| 7.5 | Efecto de la respuesta rápida | `resultados/cap7/figura_7_5_intervencion.png` | [07 §6](notebooks/07_cgmacros.ipynb) |
## Tablas

| Tabla | Contenido | Archivo | Notebook |
|---|---|---|---|
| 1.1 | Comportamiento de las hormonas reguladoras | — (sin cómputo) | — |
| 2.1 | Limitaciones del sistema de monitoreo | — (sin cómputo) | — |
| 3.1 | Exponente DFA por grupo y señal | `resultados/cap3/tabla_3_1_dfa.csv` | [03 §2](notebooks/03_analisis_de_los_datos.ipynb) |
| 3.2 | Variables y ecuaciones de S(t) | — (definiciones; implementadas en glucosa/forzamiento.py, sin cómputo) | — |
| 3.3 | Clasificación ordinal del alba | — (criterios; implementados en glucosa/detector.py, sin cómputo) | — |
| 3.4 | Exponente y tiempo de Lyapunov | `resultados/cap3/tabla_3_4_lyapunov.csv` | [03 §5](notebooks/03_analisis_de_los_datos.ipynb) |
| 3.5 | Indicador Stat_MS | `resultados/cap3/tabla_3_5_optm.csv` | [03 §7](notebooks/03_analisis_de_los_datos.ipynb) |
| 4.1 | Parámetros de cinco sujetos DBT2 | `resultados/cap4/tabla_4_1_parametros.csv` | [04 §2](notebooks/04_el_modelo.ipynb) |
| 4.2 | Métricas de validación por serie | `resultados/cap4/tabla_4_2_validacion.csv` | [04 §3](notebooks/04_el_modelo.ipynb) |
| 5.1 | Métricas de error, DBT2 | `resultados/cap5/tabla_5_1_DBT2.csv` | [05 §1](notebooks/05_analisis_del_error.ipynb) |
| 5.2 | Métricas de error, NDBT2 | `resultados/cap5/tabla_5_2_NDBT2.csv` | [05 §1](notebooks/05_analisis_del_error.ipynb) |
| 5.3 | MAE diario, DBT2 | `resultados/cap5/tabla_5_3_mae_diario_DBT2.csv` | [05 §5](notebooks/05_analisis_del_error.ipynb) |
| 5.4 | MAE diario, NDBT2 | `resultados/cap5/tabla_5_4_mae_diario_NDBT2.csv` | [05 §5](notebooks/05_analisis_del_error.ipynb) |
| 6.1 | Parámetros del sujeto 5 | `resultados/cap6/tabla_6_1_parametros_sujeto5.csv` | [06](notebooks/06_optimizacion_control_glucemico.ipynb) |
| 6.2 | Métricas del sujeto 5 | `resultados/cap6/tabla_6_2_sujeto5.csv` | [06 §3](notebooks/06_optimizacion_control_glucemico.ipynb) |
| 6.3 | Resultado poblacional | `resultados/cap6/tabla_6_3_poblacional.csv` | [06 §5](notebooks/06_optimizacion_control_glucemico.ipynb) |
| 6.4 | Transformaciones globales de S(t) | `resultados/cap6/tabla_6_4_transformaciones.csv` | [06 §6](notebooks/06_optimizacion_control_glucemico.ipynb) |
| 7.1 | Composición de la cohorte | `resultados/cap7/tabla_7_1_cohorte.csv` | [07 §1](notebooks/07_cgmacros.ipynb) |
| 7.2 | Herramientas y ecuaciones | — (mapa de referencias, sin cómputo) | — |
| 7.3 | H y C por grupo | `resultados/cap7/tabla_7_3_hxc.csv` | [07 §2](notebooks/07_cgmacros.ipynb) |
| 7.4 | Desempeño del detector | `resultados/cap7/tabla_7_4_detector_original.csv` | [07 §3.1](notebooks/07_cgmacros.ipynb) |
| 7.5 | Intensidad de los eventos | `resultados/cap7/tabla_7_5_intensidad_original.csv` | [07 §3.1](notebooks/07_cgmacros.ipynb) |
| 7.6 | Frecuencia del alba | `resultados/cap7/tabla_7_6_alba_original.csv` | [07 §3.1](notebooks/07_cgmacros.ipynb) |
| 7.7 | Descriptores reales y simulados | `resultados/cap7/tabla_7_7_modelo.csv` | [07 §5](notebooks/07_cgmacros.ipynb) |
| 7.8 | Parámetros de la respuesta rápida | `resultados/cap7/tabla_7_8_parametros_respuesta_rapida.csv` | [07 §6](notebooks/07_cgmacros.ipynb) |
| 7.9 | Resultados individuales de la intervención | `resultados/cap7/tabla_7_9_intervencion.csv` | [07 §6](notebooks/07_cgmacros.ipynb) |
