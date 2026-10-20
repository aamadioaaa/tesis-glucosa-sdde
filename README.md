# Modelización matemática de la dinámica de la glucosa en sujetos con DMT2

Código, datos y notebooks reproducibles de la tesis doctoral

> **«Modelización matemática de la dinámica de la glucosa en sujetos con Diabetes Mellitus
> Tipo II mediante ecuaciones diferenciales con retardo y modelos estocásticos»**
> Ariel Alejandro Amadio — Doctorado en Ingeniería, mención Ciencia de Datos y Procesamiento
> de Señales e Imágenes, Universidad Tecnológica Nacional (2026).
> Director: Dr. Dino Otero · Codirector: Dr. Walter Legnani.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/aamadioaaa/tesis-glucosa-sdde/HEAD?urlpath=lab/tree/notebooks)

## Para el jurado: cómo revisar los resultados

Hay tres formas de revisar los resultados, de menor a mayor esfuerzo:

1. **Ver los notebooks ya ejecutados (sin instalar nada).** Los notebooks de la carpeta
   [`notebooks/`](notebooks) se publican con sus salidas, figuras y tablas. GitHub los
   muestra directamente en el navegador. Las tablas y figuras regeneradas también están en
   [`resultados/`](resultados).
2. **Ejecutarlos en el navegador.** El botón *Binder* de arriba abre JupyterLab con todo
   instalado. También se puede usar Google Colab con los enlaces de la tabla. En Colab, la
   primera celda clona el repositorio.
3. **Ejecutarlos en una computadora propia:**

   ```bash
   git clone https://github.com/aamadioaaa/tesis-glucosa-sdde.git
   cd tesis-glucosa-sdde
   pip install -r requirements.txt
   python -m pytest -q                 # 46 pruebas de reproducibilidad (≈ 5 s)
   python scripts/ejecutar_todo.py     # re-ejecuta los 5 notebooks (≈ 10 min)
   ```

Cada notebook termina con una **tabla de verificación** que compara las cifras calculadas
con las publicadas en la tesis.

## Mapa capítulo → código

| Cap. | Tema | Notebook | Abrir |
|---|---|---|---|
| 1 | Las hormonas y el ciclo circadiano | — (teórico) | — |
| 2 | El sensor CGM y el protocolo de muestreo | — (teórico) | — |
| 3 | Los datos y su análisis: βc, DFA, S(t), alba/ingestas, Lyapunov, H×C, OPTM, Takens | [`03_analisis_de_los_datos.ipynb`](notebooks/03_analisis_de_los_datos.ipynb) | [Colab](https://colab.research.google.com/github/aamadioaaa/tesis-glucosa-sdde/blob/main/notebooks/03_analisis_de_los_datos.ipynb) |
| 4 | El modelo SDDE de Langevin con retardo: integración, calibración, validación | [`04_el_modelo.ipynb`](notebooks/04_el_modelo.ipynb) | [Colab](https://colab.research.google.com/github/aamadioaaa/tesis-glucosa-sdde/blob/main/notebooks/04_el_modelo.ipynb) |
| 5 | Análisis del error del modelo | [`05_analisis_del_error.ipynb`](notebooks/05_analisis_del_error.ipynb) | [Colab](https://colab.research.google.com/github/aamadioaaa/tesis-glucosa-sdde/blob/main/notebooks/05_analisis_del_error.ipynb) |
| 6 | Optimización del control glucémico (operador wavelet, actualización causal, TPE) | [`06_optimizacion_control_glucemico.ipynb`](notebooks/06_optimizacion_control_glucemico.ipynb) | [Colab](https://colab.research.google.com/github/aamadioaaa/tesis-glucosa-sdde/blob/main/notebooks/06_optimizacion_control_glucemico.ipynb) |
| 7 | Aplicación al conjunto público CGMacros | [`07_cgmacros.ipynb`](notebooks/07_cgmacros.ipynb) | [Colab](https://colab.research.google.com/github/aamadioaaa/tesis-glucosa-sdde/blob/main/notebooks/07_cgmacros.ipynb) |

Cada notebook regenera las figuras y tablas de su capítulo y termina con una tabla de
verificación que compara las cifras calculadas con las del texto. El Capítulo 7 ejecuta además
los scripts originales del capítulo ([`originales/cap7_cgmacros`](originales/cap7_cgmacros)).
Los parámetros de cómputo y las convenciones numéricas están en
[`REPRODUCIBILIDAD.md`](REPRODUCIBILIDAD.md).

El índice [`INDICE_FIGURAS_Y_TABLAS.md`](INDICE_FIGURAS_Y_TABLAS.md) indica, para cada figura y
tabla de la tesis, el archivo que la regenera y el notebook que la produce.

## Estructura

```
glucosa/                 paquete Python con todos los métodos
  forzamiento.py         S(t), ecs. (3.6)–(3.13)
  espectral.py           Welch, βc, DFA
  ordinal.py             Bandt–Pompe, plano H×C, OPTM, ‖v‖, índice ν
  no_lineal.py           Lyapunov (Rosenstein), información mutua, falsos vecinos
  detector.py            fenómeno del alba e ingestas, ecs. (3.14)–(3.35b)
  sdde.py                SDDE (4.3), integrador (4.4)–(4.9), función objetivo (4.23), calibración
  error.py               métricas del Cap. 5 y grilla de Clarke
  intervencion.py        starlet B3, operador (6.5), actualización causal (6.6), seguridad (6.7), TPE
  cgmacros.py            lectura y control de calidad de CGMacros, reconstrucción DCT-PCA
notebooks/               un notebook por capítulo (formato .ipynb y su par .py legible)
originales/               código original recuperado: Figuras 5.1–5.3 (cap5_figuras) y Capítulo 7 (cap7_cgmacros)
datos/
  cgm/                   registros CGM propios (15 min, 1200 muestras, anonimizados)
  modelo/                parámetros calibrados y trayectorias simuladas archivadas
  cgmacros/compacto/     versión compacta de CGMacros (CC BY-NC-SA 4.0)
resultados/              tablas (.csv) y figuras (.png) regeneradas por los notebooks
scripts/                 calibración completa, ejecución de todos los notebooks, descarga de CGMacros
tests/                   pruebas automáticas de reproducibilidad
```

## Datos

* **Registros propios** (`datos/cgm`): series de glucosa intersticial (FreeStyle Libre 2)
  cada 15 min. Contienen solo la columna `glucosa_mg_dl`, sin fechas ni identificadores
  personales.
  * `DBT2/DBT2_01…20`: registros con diagnóstico de diabetes tipo 2.
  * `NDBT2/NDBT2_01…05`: registros sin diagnóstico.
  * `NDBT2_aumentado/NDBT2_06…20`: realizaciones sintéticas (DCT + PCA + GMM). No son
    sujetos independientes.

  Durante la tesis los archivos se llamaban `subject_i_E.csv` (DBT2) y `subject_i_S.csv`
  (NDBT2).
* **Parámetros del modelo** (`datos/modelo`): resultado de la calibración del Capítulo 4,
  con precisión completa, junto con las trayectorias simuladas tal como se archivaron.
* **CGMacros** (Cap. 7): Gutierrez-Osuna, R., Kerr, D., Mortazavi, B. & Das, A. (2025).
  *CGMacros: a scientific dataset for personalized nutrition and diet monitoring* (v1.0.0).
  PhysioNet. https://doi.org/10.13026/3z8q-x658 — licencia CC BY-NC-SA 4.0. El repositorio
  incluye una versión compacta (lecturas Libre, marcas de comidas y HbA1c). Los archivos
  originales se descargan con `python scripts/descargar_cgmacros.py`.

## Licencias

* Código: [MIT](LICENSE).
* Datos propios (`datos/cgm`, `datos/modelo`): [CC BY 4.0](datos/LICENCIA.md).
* Datos derivados de CGMacros (`datos/cgmacros`): CC BY-NC-SA 4.0, con la atribución indicada
  arriba.

## Cómo citar

Ver [`CITATION.cff`](CITATION.cff).
