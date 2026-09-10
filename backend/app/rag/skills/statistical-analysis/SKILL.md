---
name: statistical-analysis
description: >-
  Usa esta skill cuando el usuario pida un análisis estadístico sobre datos
  numéricos: medias, correlaciones, pruebas de hipótesis o regresiones.
---

# Análisis Estadístico

Procedimiento para elegir y ejecutar la operación correcta con la herramienta
`statistics`.

## 1. Identifica los datos

Extrae la(s) serie(s) numérica(s) de la pregunta o de la evidencia recuperada con
`pdf_search`. Nunca inventes valores: si faltan datos para completar el análisis,
dilo explícitamente en la Final Answer.

## 2. Elige la operación

- Resumen de una sola serie → `mean`, `median`, `std`, `variance`, `min`, `max`,
  `quartiles`, `iqr`.
- Relación entre dos series → `correlation_pearson` (relación lineal) o
  `correlation_spearman` (relación monótona, datos no normales o con outliers).
- Comparar una muestra contra un valor esperado → `t_test_one_sample` (requiere
  `sample_mean`).
- Comparar dos muestras → `t_test_independent` (grupos distintos) o
  `t_test_paired` (mismos sujetos, mediciones repetidas).
- Predecir una variable a partir de otra → `linear_regression` (requiere `x` e `y`).

## 3. Ejecuta y reporta con cautela

Llama a `statistics` con los parámetros correctos para la operación elegida. La
herramienta ya añade advertencias automáticas para muestras pequeñas (n < 10) y
para pruebas con supuestos distribucionales no verificados — inclúyelas siempre en
tu respuesta, no las omitas ni las suavices.

## 4. Cita el origen de los datos

Si los datos provienen de un documento, cita la fuente `[D#]` de la que se
extrajeron antes de presentar el resultado numérico. Si provienen únicamente de lo
que escribió el usuario en la pregunta, dilo así en vez de atribuirlos a una fuente
que no existe.
