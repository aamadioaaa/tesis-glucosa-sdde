# Script original de las Figuras 5.1–5.3

`capitulo_5_figuras.py` es el script con el que se diseñaron las Figuras 5.1, 5.2 y 5.3 del
Capítulo 5 (paneles A–I). Se publica tal como estaba, con un único cambio: el modo de datos
y las carpetas de entrada y salida se pueden fijar con variables de entorno (`MODO_DATOS`,
`CAP5_DATOS`, `CAP5_FIGURAS`). `LEEME_original.md` es su documentación original.

El script tiene dos modos. En «sustituto» reconstruye señales compatibles con las tablas,
sólo para revisar el diseño; en «real» lee, para cada serie, la glucosa medida (GE) y la
simulada (GS). `reproducir.py` arma esos archivos a partir de `datos/` y lo ejecuta en modo
«real»:

```bash
python originales/cap5_figuras/reproducir.py
```

Las figuras (PNG y PDF vectorial) quedan en `resultados/cap5/original/`. El notebook 05 lo
ejecuta en su §7.
