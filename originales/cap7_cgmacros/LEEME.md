# Código original del Capítulo 7 (CGMacros)

Esta carpeta contiene el paquete de reproducibilidad que se armó durante la escritura del
Capítulo 7 (`CGMacros_reproducibilidad.zip`). Se recuperó de los archivos de trabajo del
autor en octubre de 2026 y se publica tal como estaba. Las únicas modificaciones son las
siguientes:

* Las rutas absolutas del equipo original se reemplazaron por dos variables
  (`_rutas.py`): `CGMACROS_DIR` (datos) y `SALIDA_CAP7_ORIGINAL` (salidas). Cada script
  tiene una primera línea agregada que las importa. El resto del código no se tocó.
* Se agregó `reproducir.py`, que corre los scripts y arma las Tablas 7.4–7.6 con el
  agrupamiento final de la tesis (ver más abajo).

## Contenido

| Archivo | Qué hace |
|---|---|
| `extract_St.py` | S(t) por sujeto a partir de la serie Libre remuestreada a 15 min |
| `consolidar_3grupos.py` | detector de ingestas y del fenómeno del alba, plano H×C, agregados por grupo |
| `detect_diabeticos.py`, `detect_sanos.py` | versiones del detector por grupo, con el detalle de cada ventana |
| `plano_HC_libre_full.py` | plano H×C con las fronteras |
| `sim_langevin.py` | modelo de Langevin y calibración (extraído de `Simulador_DBT2.ipynb`) |
| `run_sim_cgm.py` | calibra el modelo en los sujetos de `cgm_diab_in/` |
| `grid_diab.py`, `plot_diab_burgers.py` | figuras |
| `cgm_diab_in/` | series de glucosa a 15 min de los 18 sujetos con DT2 según el paquete (entrada de `run_sim_cgm.py`) |
| `archivado/` | salidas tal como se archivaron: tablas, trayectorias simuladas, series S(t) |
| `figuras_archivadas/` | figuras originales del paquete |
| `README_original.md` | README original del paquete |

## Cómo se llega a las cifras de la tesis

```bash
python originales/cap7_cgmacros/reproducir.py
```

1. Si no están los CSV originales de CGMacros, `reproducir.py` los reconstruye a partir de la
   versión compacta del repositorio. Los scripts sólo leen `Timestamp`, `Libre GL`, `Meal Type`
   y `bio.csv`.
2. Se ejecutan `extract_St.py`, `consolidar_3grupos.py`, `detect_diabeticos.py` y `plot_diab_burgers.py` sin cambios.
3. El paquete original agrupaba por HbA1c **o** glucemia en ayunas y usaba los 45 sujetos
   (`archivado/agg_3grupos.csv`). La versión final del capítulo agrupa sólo por HbA1c
   (< 5,7 / 5,7–6,4 / ≥ 6,5 %) y excluye 007 (menos de 6 días utilizables) y 018 (más de 24 h
   reconstruidas). Con ese agrupamiento se obtienen exactamente las Tablas 7.4 a 7.6.
4. Las cifras del ajuste del modelo (Sec. 7.7: ρ = 0,739, RMSE = 24,8 mg/dL, TIR 82 → 81 %)
   son los promedios de `archivado/sim_CGM_diab_resumen.csv` sobre los 14 sujetos con
   HbA1c ≥ 6,5 %. La corrida no puede repetirse bit a bit, porque `sim_langevin.py` siembra el
   ruido con `hash(sujeto)` y Python aleatoriza ese valor en cada sesión.

**Versión de PyWavelets.** `requirements.txt` fija PyWavelets 1.7: desde la versión 1.8 la
wavelet `cmor1.5-1.0` cambió ligeramente.

## Licencia

Código: MIT, como el resto del repositorio. Las series de `cgm_diab_in/` y `archivado/`
derivan de CGMacros (PhysioNet, CC BY-NC-SA 4.0) y se distribuyen bajo esa licencia.
