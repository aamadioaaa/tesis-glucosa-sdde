# CGMacros — versión compacta

Adaptación de **CGMacros v1.0.0** (PhysioNet) con las columnas necesarias para el Capítulo 7:

* `libre_1min.csv.gz` — `sujeto, timestamp, libre_gl`: filas publicadas del sensor Abbott
  FreeStyle Libre Pro (grilla de 1 min interpolada por los autores).
* `comidas.csv` — `sujeto, timestamp, tipo`: marcas temporales de las ingestas registradas.
* `bio.csv` — `subject, A1c PDL (Lab), Fasting GLU - PDL (Lab)`: HbA1c y glucemia en ayunas
  de laboratorio (la HbA1c asigna el grupo; la glucemia en ayunas la usa el código original
  del capítulo, ver `originales/cap7_cgmacros`).

Generada con `scripts/preparar_cgmacros_compacto.py` a partir de los CSV originales. Las
fechas son las publicadas por PhysioNet (desplazadas por los autores).

**Fuente y licencia.** Gutierrez-Osuna, R., Kerr, D., Mortazavi, B., & Das, A. (2025).
*CGMacros: a scientific dataset for personalized nutrition and diet monitoring* (version
1.0.0). PhysioNet. https://doi.org/10.13026/3z8q-x658. Las fuentes piden citar además el
artículo asociado en *Scientific Data* (Das et al., 2025).

Licencia: Creative Commons Atribución-NoComercial-CompartirIgual 4.0 Internacional
(CC BY-NC-SA 4.0) — https://creativecommons.org/licenses/by-nc-sa/4.0/
