# OCR Implementation Guide

## Overview

El sistema PDF Assistant RAG ahora incluye capacidades completas de **OCR (Optical Character Recognition)** para extraer texto de documentos PDF basados en imágenes, incluyendo archivos escaneados y PDFs de imagen pura.

## Características

### ✅ Soporte Completo para PDFs de Imagen

- **Detección automática**: El sistema detecta automáticamente cuándo falla la extracción de texto tradicional
- **Fallback automático**: Cuando otros métodos no extraen suficiente texto, OCR se invoca automáticamente
- **Cadena de extracción robusta**: Intenta múltiples métodos de extracción en orden:
  1. Unstructured (extrae tablas complejas)
  2. pdfplumber (detecta tablas y formato)
  3. PyMuPDF (extracción básica)
  4. **OCR (pytesseract + pdf2image)** - último recurso para PDFs de imagen

### 🔧 Configuración

#### Dependencias Requeridas

Las siguientes dependencias se agregan automáticamente a `backend/requirements.txt`:

```txt
pdf2image    # Convierte páginas PDF a imágenes
pytesseract  # Motor OCR de Tesseract
pillow       # Procesamiento de imágenes
```

#### Instalación Manual

Si necesitas instalar las dependencias manualmente:

```bash
pip install pdf2image pytesseract pillow
```

**Nota sobre Tesseract**: `pytesseract` es un wrapper de Python para Tesseract OCR.
En Linux/Debian, instala:

```bash
sudo apt-get install tesseract-ocr tesseract-ocr-spa tesseract-ocr-eng
```

En macOS con Homebrew:

```bash
brew install tesseract
```

En Windows, descarga el instalador desde: https://github.com/UB-Mannheim/tesseract/wiki

### 📋 Características Técnicas

#### Función Principal: `extract_pdf_with_ocr()`

```python
def extract_pdf_with_ocr(filepath: str) -> List[Dict[str, Any]]:
    """
    Extract text from image-based PDFs using OCR (Tesseract via pdf2image).
    
    Returns:
        List of dicts with keys:
        - 'text': Texto extraído via OCR
        - 'page': Número de página
        - 'chunk_type': Siempre 'text'
        - 'ocr_source': True (indica que vino de OCR)
    """
```

#### Parámetros OCR

- **DPI**: 200 (configurable) - Balance entre calidad y velocidad
- **Formato**: PPM (Portable PixMap) - formato eficiente para conversión
- **Idiomas**: spa+eng (Español e Inglés) - configurable en `extract_pdf_with_ocr()`

#### Manejo de Errores

La implementación es **altamente robusta**:

✅ **Dependencias faltantes**: Retorna lista vacía sin fallar
✅ **PDF corrupto**: Maneja gracefully con logging de warnings
✅ **Páginas vacías**: Salta páginas donde OCR no extrae texto
✅ **Errores de conversion**: Continúa con siguientes páginas

### 🧪 Pruebas Incluidas

Se incluyen 5 pruebas completas para validar la funcionalidad:

```bash
# Ejecutar todas las pruebas OCR
pytest backend/tests/test_chunker.py -k "ocr" -v

# Pruebas específicas:
# 1. test_ocr_extraction_with_mock_pdf2image_and_pytesseract
#    - Valida extracción básica de OCR
# 2. test_ocr_fallback_when_other_methods_fail
#    - Valida que OCR se usa como fallback
# 3. test_ocr_handles_missing_dependencies
#    - Valida comportamiento sin dependencias
# 4. test_ocr_handles_corrupted_pdf
#    - Valida manejo de PDFs corruptos
# 5. test_ocr_skips_empty_results
#    - Valida que salta páginas sin texto
```

### 📊 Flujo de Procesamiento

```
Usuario sube PDF (escaneado o basado en imagen)
           ↓
FastAPI valida el archivo
           ↓
Archivo guardado en disco
           ↓
Tarea enviada a Celery Worker
           ↓
extract_pdf() intenta:
  1. Unstructured → extrae, o continúa
  2. pdfplumber  → extrae, o continúa  
  3. PyMuPDF     → extrae, o continúa
  4. OCR (AQUÍ)  → ✅ Extrae de imágenes
           ↓
Resultado devuelto a Chunker
           ↓
Chunking → Embeddings → Vector Storage
           ↓
Documento listo para RAG queries
```

### 🎯 Casos de Uso

#### Caso 1: PDF Escaneado Simple

```python
# Un libro digitalizado por scanner
from app.rag.chunker import chunk_document

chunks = chunk_document("scanned_book.pdf")
# Sistema automáticamente:
# 1. Intenta métodos tradicionales
# 2. Falla porque es imagen
# 3. Invoca OCR
# 4. Extrae todo el texto
```

#### Caso 2: PDF Mixto (Texto + Imagen)

```python
# Documento con algunas páginas digitales y algunas escaneadas
chunks = chunk_document("mixed_document.pdf")
# Sistema extrae:
# - Páginas 1-5: Texto digital (métodos rápidos)
# - Páginas 6-10: Imágenes (OCR automático)
```

#### Caso 3: Monitoreo y Logging

```python
# El sistema registra automáticamente:
# WARNING: "Unstructured extraction failed, falling back"
# WARNING: "pdfplumber extraction failed, falling back"
# WARNING: "PyMuPDF extraction failed, falling back to OCR"
# INFO: "Successfully extracted text from file.pdf using OCR"
```

### 🔍 Verificación de Funcionamiento

#### En Producción

1. **Sube un PDF escaneado** via UI
2. **Observa el log de procesamiento**:
   ```
   Processing document...
   Status: processing
   Stage: extracting
   ```
3. **Una vez completado**:
   - El documento aparecerá en la lista
   - Los chunks estarán disponibles para búsqueda
   - Las queries RAG pueden usar el contenido extraído

#### Testing Local

```bash
# Test básico
pytest backend/tests/test_chunker.py::test_ocr_extraction_with_mock_pdf2image_and_pytesseract -v

# Test de fallback
pytest backend/tests/test_chunker.py::test_ocr_fallback_when_other_methods_fail -v

# Todos los tests
pytest backend/tests/test_chunker.py -v
```

### ⚙️ Configuración Avanzada

#### Cambiar Idiomas OCR

Abre `backend/app/rag/chunker.py` y modifica la línea en `extract_pdf_with_ocr()`:

```python
# Actual
text = pytesseract.image_to_string(image, lang='spa+eng')

# Para solo español
text = pytesseract.image_to_string(image, lang='spa')

# Para múltiples idiomas
text = pytesseract.image_to_string(image, lang='spa+eng+fra+deu')
```

#### Cambiar Resolución DPI

En `extract_pdf_with_ocr()`:

```python
# Actual (200 DPI - balance)
images = convert_from_path(filepath, dpi=200, fmt='ppm')

# Para mejor calidad (más lento)
images = convert_from_path(filepath, dpi=300, fmt='ppm')

# Para mayor velocidad (menos preciso)
images = convert_from_path(filepath, dpi=150, fmt='ppm')
```

### 📈 Performance

| Escenario | Velocidad | Calidad OCR |
|-----------|-----------|------------|
| DPI 150   | Rápido    | Buena     |
| DPI 200   | Moderado  | Muy Buena |
| DPI 300   | Lento     | Excelente |

**Recomendación**: DPI 200 ofrece el mejor balance

### 🚨 Troubleshooting

#### Error: `pytesseract not available`

**Solución**:
```bash
pip install pytesseract
# Linux
sudo apt-get install tesseract-ocr
```

#### Error: `libGL.so.1 not found` (en Docker)

**Solución**: Las imágenes Docker del proyecto ya incluyen las dependencias necesarias. Si compilas tu propia imagen:

```dockerfile
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-spa \
    tesseract-ocr-eng
```

#### OCR retorna texto vacío

**Posibles causas**:
1. PDF realmente no tiene texto legible
2. Resolución muy baja
3. Idioma no instalado en Tesseract

**Soluciones**:
- Aumentar DPI a 300
- Instalar paquete de idioma: `sudo apt-get install tesseract-ocr-spa`
- Verificar calidad del scan original

#### Performance lento en PDFs grandes

**Solución**: Aumentar paralelización en Celery Worker:

```bash
# En lugar de ejecutar un worker
celery -A app.celery_app worker --loglevel=info --concurrency=2

# Ejecutar con más workers
celery -A app.celery_app worker --loglevel=info --concurrency=4
```

### 📚 Referencias

- [pytesseract Documentation](https://github.com/madmaze/pytesseract)
- [pdf2image Documentation](https://github.com/Belval/pdf2image)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Langcodes para OCR](https://github.com/tesseract-ocr/tessdata/blob/main/LANGUAGE_DATA.md)

### ✨ Próximas Mejoras

Funcionalidades futuras consideradas:

- [ ] Soporte para OCR multilingüe automático
- [ ] Detección de orientación de página
- [ ] Preprocessing de imagen (binarización, deesquew)
- [ ] Caching de resultados OCR
- [ ] Métricas de confianza OCR por página
- [ ] Admin endpoint para re-OCR de documentos
- [ ] Soporte para otros motores OCR (EasyOCR, Paddle OCR)

---

**Última actualización**: 2026-07-02
**Versión OCR**: 1.0
