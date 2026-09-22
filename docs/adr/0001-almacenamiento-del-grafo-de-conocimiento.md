# ADR 0001 — Almacenamiento del grafo de conocimiento: networkx + JSON, no Neo4j

- **Estado:** aceptado
- **Fecha:** 2026-09-21
- **Afecta a:** `backend/app/rag/graph_builder.py`, `backend/app/rag/graph_retriever.py`

## Contexto

ATLAS construye un grafo de conocimiento por documento (GraphRAG): extrae entidades con spaCy y crea aristas por co-ocurrencia dentro del mismo fragmento, ponderadas y anotadas con las páginas donde aparecen. Ese grafo se usa para añadir contexto de relaciones a la recuperación (`get_entity_context`), no para responder por sí mismo.

Durante la limpieza de dependencias de septiembre de 2026 se detectó una discrepancia: `neo4j>=5.0` estaba declarado en `backend/requirements.txt` pero **no se importaba en ninguna línea del código**, y ningún otro paquete lo requería. La implementación real siempre ha usado `networkx` en memoria, persistido como JSON (`nx.node_link_data`) en un archivo por par usuario/documento bajo `GRAPH_PERSIST_DIR`.

Había entonces que decidir entre migrar a Neo4j de verdad o dejar constancia de por qué no.

### Cómo se consulta realmente el grafo

Este es el dato determinante. `get_entity_context` hace exactamente esto:

1. Empareja entidades de la consulta contra los nodos del grafo.
2. Para cada nodo emparejado, recorre **solo sus vecinos inmediatos** (`graph.neighbors`).
3. Agrega las relaciones por peso y devuelve las `GRAPH_MAX_RELATIONSHIPS` más fuertes como texto.

No hay recorridos de varios saltos, ni búsqueda de caminos, ni detección de comunidades, ni consultas de patrones. Es una expansión de un salto sobre una lista de adyacencia.

### Tamaño real del corpus actual

Medido sobre la instalación de desarrollo (31 documentos procesados):

| Métrica | Valor |
| --- | --- |
| Grafos persistidos | 31 |
| Ocupación en disco | 21.1 MB |
| Nodos (total / mediana / máximo) | 7 067 / 135 / 1 234 |
| Aristas (total / mediana / máximo) | 111 095 / 1 564 / 15 491 |
| Archivo por grafo (mediana / máximo) | 301 KB / 2.9 MB |
| Consulta acotada a un documento | carga un solo archivo |
| Consulta a todo el corpus | ~0.65 s en caliente |

## Decisión

Se mantiene **networkx en memoria con persistencia JSON por usuario/documento**, y se elimina la dependencia `neo4j` no utilizada.

El motivo es que Neo4j aporta su valor en exactamente lo que este sistema no hace: recorridos de varios saltos, Cypher, consultas de patrones sobre un grafo global y algoritmos de grafo. Pagar por un servicio con estado —otro contenedor, otro backup, otra migración, otro modo de fallo— para atender consultas de un salto sobre grafos cuya mediana es de 135 nodos sería coste sin contrapartida.

Pesa además la restricción de la máquina de desarrollo: una MacBook Pro M1 de 8 GB donde Ollama ya es el proceso dominante. El perfil local evita Docker deliberadamente, así que una base de datos de grafos obligatoria empujaría el desarrollo local hacia una infraestructura que esa máquina no sostiene cómodamente.

## Consecuencias

### A favor

- Cero servicios adicionales: el grafo es un archivo, se respalda copiando un directorio y se borra con el documento.
- Aislamiento por usuario y documento **por construcción**, no por una cláusula de filtrado que alguien pueda olvidar. Cada grafo vive en su propio archivo con su propio `user_id` en los metadatos.
- Las pruebas no necesitan servidor: se escribe un JSON y se lee.
- Una lectura de grafo corrupta o ausente degrada a "sin contexto de grafo" y nunca rompe una respuesta.

### En contra, y esto es lo que hay que vigilar

- **Una consulta a todo el corpus carga todos los grafos del usuario.** `_candidate_graphs` con `document_id=None` lee y deserializa cada archivo en cada consulta. Hoy son ~0.65 s con 31 documentos, y el coste **crece linealmente** con el número de documentos del usuario, no con la relevancia de estos.
- No existe un índice: emparejar la consulta con los nodos implica recorrerlos.
- No hay consultas entre documentos de verdad. Las relaciones de distintos documentos se agregan en memoria por coincidencia de nombre de entidad, no por identidad de entidad resuelta.
- El formato depende de la versión de networkx: la clave de las aristas en `node_link_data` pasó de `links` a `edges` en versiones recientes. Se lee y escribe con la misma versión, así que es consistente, pero cualquier herramienta externa que lea estos JSON debe tenerlo en cuenta.

## Cuándo habría que revisar esta decisión

Migrar a un motor de grafos cuando se cumpla **cualquiera** de estas condiciones, no antes:

1. La consulta al corpus completo supera de forma sostenida ~2 s. Con el coste lineal medido, eso ocurre alrededor de los **100 documentos por usuario**.
2. Aparece un caso de uso real de varios saltos: "qué conecta A con C a través de B", caminos, centralidad o detección de comunidades. Hoy no existe ninguno.
3. Se necesita resolución de entidades entre documentos (que "NOx" y "óxidos de nitrógeno" sean el mismo nodo), que es donde un grafo global con identidad propia empieza a rendir.
4. Varios procesos deben escribir el mismo grafo a la vez. El modelo de un archivo por documento asume un único escritor.

El paso intermedio antes de saltar a Neo4j sería mover los grafos a las tablas de PostgreSQL que el perfil de producción ya usa, lo que resolvería los puntos 1 y 4 sin añadir un servicio nuevo.

## Alternativas descartadas

| Alternativa | Por qué no |
| --- | --- |
| Neo4j | Su valor está en recorridos y Cypher, que este sistema no usa; añade un servicio con estado |
| SQLite con tablas de aristas | Resolvería la carga completa, pero ninguna métrica actual lo justifica todavía |
| Grafo global único por usuario | Rompería el aislamiento por documento, que hoy es estructural |
| Mantener `neo4j` declarado "por si acaso" | Una dependencia no usada es deuda: sugiere una arquitectura que no existe |
