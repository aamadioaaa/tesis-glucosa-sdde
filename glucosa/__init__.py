"""
glucosa — código de la tesis doctoral
«Modelización matemática de la dinámica de la glucosa en sujetos con
Diabetes Mellitus Tipo II mediante ecuaciones diferenciales con retardo y
modelos estocásticos» (A. Amadio, UTN FRGP, 2026).

Módulos
-------
datos         carga de registros CGM, parámetros calibrados y CGMacros
forzamiento   indicador de forzamiento metabólico S(t) (Cap. 3, ec. 3.6–3.13)
espectral     PSD de Welch, exponente βc, DFA (Cap. 3)
ordinal       Bandt–Pompe, plano H×C, OPTM y norma del vector estacionario (Cap. 3)
no_lineal     exponente de Lyapunov (nolds), información mutua, falsos vecinos (Cap. 3)
sdde          SDDE de Langevin con retardo, integrador y calibración (Cap. 4)
error         métricas de error y grilla de Clarke (Cap. 5)
detector      detección de fenómeno del alba e ingestas (Cap. 3, Sec. 3.3.6)
intervencion  operador wavelet no decimado, actualización causal y seguridad (Caps. 6–7)
"""
__version__ = "1.0.0"
