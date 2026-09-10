---
name: code-review
description: >-
  Usa esta skill cuando el usuario pida revisar, auditar, depurar, evaluar la calidad
  de código del repositorio o de un fragmento que haya compartido, o generar código
  nuevo cuya lógica deba seguir un documento cargado por el usuario.
---

# Revisión y Generación de Código (Percibe-Razona-Actúa)

Procedimiento para producir revisiones de código y código nuevo consistentes y
verificables, en un ciclo iterativo de Percepción, Razonamiento y Actuación.

## Percibe

Reúne evidencia real antes de opinar — nunca inventes código, archivos o comportamiento:

- Si el usuario referencia un archivo del repositorio, pásalo a `code_review` mediante
  `file_path` (ruta relativa, p. ej. `backend/app/rag/tools.py`).
- Si el usuario pega un fragmento directamente, pásalo mediante `code`.
- Si hay herramientas de archivos de solo lectura disponibles (`read_file`,
  `list_directory`, `grep_file`), úsalas para inspeccionar más contexto en lugar de
  asumirlo.
- Si el código debe seguir una especificación, ecuación o comportamiento descrito en
  un documento que el usuario cargó, usa `pdf_search` antes de generar o revisar el
  código.
- Si no tienes ni archivo, ni fragmento, ni documento relevante, dilo en tu Final
  Answer en lugar de inventar contexto.

## Razona

Con lo percibido, decide el enfoque antes de actuar:

- Elige `focus` para `code_review` según lo que pida el usuario: `bugs` (por
  defecto), `seguridad`, `rendimiento`, `pruebas`.
- Si `pdf_search` devolvió evidencia relevante, decide qué parte de esa evidencia
  debe gobernar la lógica del código que vas a generar o ajustar.
- Prioriza los hallazgos aproximadamente en este orden: errores que impiden
  ejecutar el programa, bugs de resultado incorrecto, vulnerabilidades de
  seguridad, pérdida/corrupción de datos, problemas de rendimiento, arquitectura o
  mantenibilidad, calidad/legibilidad, estilo.
- Distingue problemas confirmados de hipótesis; si falta información para concluir
  algo, dilo explícitamente en vez de suponer.
- Prefiere cambios mínimos que preserven el comportamiento y las convenciones
  existentes.

## Actúa

- Llama a `code_review` con `query`, (`file_path` o `code`), `language` y `focus`
  cuando la tarea sea revisar código existente. No ejecutes el código y no asumas
  contexto que no esté en el fragmento entregado.
- Cuando la tarea sea generar o refactorizar código, colócalo directamente en tu
  Final Answer como bloque de código; si se basó en un documento cargado, menciona
  brevemente (de forma informal, no como cita obligatoria) qué fragmento lo
  respalda.
- Este modo es de solo lectura/asesoría: nunca afirmes haber escrito, modificado o
  guardado un archivo. El usuario aplica el código manualmente.
- Nunca afirmes haber ejecutado una prueba o verificación que no ejecutaste
  realmente.

## Reporta

Para una revisión, estructura la Final Answer como:

1. **Hallazgos**, clasificados por severidad (Crítico, Medio, Estilo).
2. **Evidencia**: la línea o el fragmento afectado.
3. **Recomendación** concreta por hallazgo.

Para código generado o corregido, explica brevemente los cambios y entrega el
código completo o el fragmento modificado.

Si estás en el modo dedicado de revisión de código, tu propuesta de Python se
verifica automáticamente por sintaxis entre rondas: si la ronda anterior tuvo
errores, se te mostrarán al inicio de esta ronda — corrígelos en vez de repetir la
misma respuesta. El código en otros lenguajes no se verifica automáticamente.
