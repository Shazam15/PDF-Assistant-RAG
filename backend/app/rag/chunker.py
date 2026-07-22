"""
Smart document chunking using LangChain's RecursiveCharacterTextSplitter.
Supports PDF, DOCX, TXT, and Markdown files with page-level metadata.
"""
import json
import re
import fitz  # PyMuPDF
import docx
import logging
import os
import shutil
from typing import Any, Callable, Dict, List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

CODE_EXTENSIONS = {"py", "js", "ts", "tsx", "java", "cpp", "c", "cs", "go", "rs", "sql", "ipynb"}
OCR_MIN_TEXT_CHARS_PER_PAGE = 40
ProgressCallback = Callable[[str, Optional[int], Optional[int]], None]


def _notify_progress(
    callback: Optional[ProgressCallback],
    stage: str,
    current: Optional[int] = None,
    total: Optional[int] = None,
) -> None:
    if callback:
        callback(stage, current, total)


def _token_length(text: str) -> int:
    """Fast model-independent token estimate used to bound retrieval context."""
    return len(re.findall(r"\w+|[^\w\s]", text or "", flags=re.UNICODE))


def _heading_candidate(text: str) -> str | None:
    """Detect layout-neutral section headings without domain vocabularies."""
    first_line = next((line.strip() for line in (text or "").splitlines() if line.strip()), "")
    if not first_line or len(first_line) > 120 or len(first_line.split()) > 14:
        return None
    numbered = bool(re.match(r"^(?:\d+(?:\.\d+)*|[IVXLC]+)[.)]?\s+", first_line, re.IGNORECASE))
    upper = any(char.isalpha() for char in first_line) and first_line.upper() == first_line
    title = first_line.istitle() and not first_line.endswith(('.', ';', ':'))
    return first_line if numbered or upper or title else None


def _annotate_hierarchy(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach section and parent context while keeping child chunks as retrieval units."""
    if not chunks:
        return chunks

    section_number = 0
    parent_number = 0
    current_section = "Document"
    current_parent: List[Dict[str, Any]] = []
    current_parent_tokens = 0

    def flush_parent() -> None:
        nonlocal parent_number, current_parent, current_parent_tokens
        if not current_parent:
            return
        parent_id = f"P{parent_number}"
        parent_text = "\n\n".join(str(item.get("text") or "").strip() for item in current_parent).strip()
        page_start = min(int(item.get("page") or 1) for item in current_parent)
        page_end = max(int(item.get("page") or 1) for item in current_parent)
        for item in current_parent:
            item["parent_id"] = parent_id
            item["parent_text"] = parent_text
            item["page_start"] = page_start
            item["page_end"] = page_end
        parent_number += 1
        current_parent = []
        current_parent_tokens = 0

    for chunk in chunks:
        heading = _heading_candidate(str(chunk.get("text") or ""))
        if heading and heading != current_section:
            flush_parent()
            current_section = heading
            section_number += 1

        chunk_tokens = _token_length(str(chunk.get("text") or ""))
        if current_parent and current_parent_tokens + chunk_tokens > settings.PARENT_CHUNK_SIZE:
            flush_parent()
        chunk["section_id"] = f"S{section_number}"
        chunk["section_title"] = current_section
        chunk["section"] = current_section
        chunk["token_count"] = chunk_tokens
        current_parent.append(chunk)
        current_parent_tokens += chunk_tokens

    flush_parent()
    return chunks


def _is_word_inside_bbox(word: Dict[str, Any], bbox: tuple) -> bool:
    """Return True when the word center falls inside a pdfplumber bbox."""
    x0, top, x1, bottom = bbox
    word_x = (float(word["x0"]) + float(word["x1"])) / 2
    word_y = (float(word["top"]) + float(word["bottom"])) / 2
    return x0 <= word_x <= x1 and top <= word_y <= bottom


def _words_to_text(words: List[Dict[str, Any]], line_tolerance: float = 3.0) -> str:
    """Rebuild readable text from positioned pdfplumber words."""
    if not words:
        return ""

    sorted_words = sorted(words, key=lambda item: (round(float(item["top"]) / line_tolerance), item["x0"]))
    lines: List[List[Dict[str, Any]]] = []

    for word in sorted_words:
        if not lines:
            lines.append([word])
            continue

        current_top = sum(float(item["top"]) for item in lines[-1]) / len(lines[-1])
        if abs(float(word["top"]) - current_top) <= line_tolerance:
            lines[-1].append(word)
        else:
            lines.append([word])

    text_lines = [
        " ".join(item["text"] for item in sorted(line, key=lambda item: item["x0"]))
        for line in lines
    ]
    return "\n".join(line for line in text_lines if line.strip())


def _clean_table_cell(cell: Any) -> str:
    """Normalize extracted table cell text for Markdown serialization."""
    if cell is None:
        return ""
    return re.sub(r"\s+", " ", str(cell)).strip().replace("|", "\\|")


def _table_to_markdown(rows: List[List[Any]]) -> str:
    """Serialize extracted table rows into Markdown for retrieval."""
    cleaned_rows = [
        [_clean_table_cell(cell) for cell in row]
        for row in rows
        if row and any(_clean_table_cell(cell) for cell in row)
    ]
    if not cleaned_rows:
        return ""

    width = max(len(row) for row in cleaned_rows)
    normalized = [row + [""] * (width - len(row)) for row in cleaned_rows]

    def fmt(row: List[str]) -> str:
        return "| " + " | ".join(row) + " |"

    header = normalized[0]
    separator = ["---"] * width
    body = normalized[1:]
    return "\n".join([fmt(header), fmt(separator), *[fmt(row) for row in body]])


def _page_has_enough_text(page_items: List[Dict[str, Any]]) -> bool:
    """Return True when extracted page text looks useful enough to skip OCR."""
    text = "\n".join(
        item.get("text", "")
        for item in page_items
        if item.get("chunk_type", "text") == "text"
    )
    return len(re.sub(r"\s+", "", text)) >= OCR_MIN_TEXT_CHARS_PER_PAGE


def _merge_ocr_for_sparse_pages(
    filepath: str,
    pages: List[Dict[str, Any]],
    progress_callback: Optional[ProgressCallback] = None,
) -> List[Dict[str, Any]]:
    """OCR only pages where normal extraction produced no meaningful text."""
    if not pages:
        if progress_callback:
            return extract_pdf_with_ocr(filepath, progress_callback=progress_callback)
        return extract_pdf_with_ocr(filepath)

    doc = None
    try:
        doc = fitz.open(filepath)
        page_count = len(doc)
    except Exception as exc:
        logger.warning(f"Could not inspect PDF page count before OCR merge: {exc}")
        return pages
    finally:
        if doc:
            doc.close()

    pages_by_num: Dict[int, List[Dict[str, Any]]] = {}
    for page_data in pages:
        pages_by_num.setdefault(page_data.get("page", 1), []).append(page_data)

    sparse_page_numbers = [
        page_num
        for page_num in range(1, page_count + 1)
        if not _page_has_enough_text(pages_by_num.get(page_num, []))
    ]
    if not sparse_page_numbers:
        return pages

    try:
        if progress_callback:
            ocr_pages = extract_pdf_with_ocr(
                filepath,
                page_numbers=sparse_page_numbers,
                progress_callback=progress_callback,
            )
        else:
            ocr_pages = extract_pdf_with_ocr(filepath, page_numbers=sparse_page_numbers)
    except Exception as exc:
        logger.warning(f"OCR merge failed for sparse PDF pages: {exc}")
        return pages

    if not ocr_pages:
        return pages

    merged_pages = [
        page_data
        for page_data in pages
        if page_data.get("page") not in {ocr_page["page"] for ocr_page in ocr_pages}
        or page_data.get("chunk_type") == "table"
    ]
    merged_pages.extend(ocr_pages)
    return sorted(merged_pages, key=lambda item: (item.get("page", 1), item.get("table_index", -1)))


def _resolve_tesseract_languages(pytesseract_module: Any, requested_languages: str) -> str:
    """Use only OCR languages installed in the local Tesseract data path."""
    requested = [lang.strip() for lang in requested_languages.split("+") if lang.strip()]
    try:
        installed = set(pytesseract_module.get_languages(config=""))
    except Exception as exc:
        logger.warning(f"Could not inspect installed Tesseract languages: {exc}")
        return "eng"

    available = [lang for lang in requested if lang in installed]
    if available:
        return "+".join(available)
    if "eng" in installed:
        return "eng"
    return installed.pop() if installed else "eng"


def _pdf_requires_quality_extraction(filepath: str, pages: List[Dict[str, Any]]) -> bool:
    """Detect sparse text or real tables before paying the Docling cost."""
    try:
        doc = fitz.open(filepath)
    except Exception:
        return True

    try:
        populated_pages = {
            int(item.get("page") or 1)
            for item in pages
            if str(item.get("text") or "").strip()
        }
        if len(populated_pages) < len(doc):
            return True
        for page_index in range(min(len(doc), 20)):
            page = doc[page_index]
            finder = getattr(page, "find_tables", None)
            if finder is None:
                continue
            try:
                if getattr(finder(), "tables", None):
                    return True
            except Exception:
                continue
        return False
    finally:
        doc.close()


def extract_pdf(
    filepath: str,
    progress_callback: Optional[ProgressCallback] = None,
) -> List[Dict[str, Any]]:
    """Extract PDF text while preserving tables as separate chunks.

    Prefer stable Python PDF extractors by default. Unstructured can be enabled
    with PDF_USE_UNSTRUCTURED=true for advanced table extraction, but it is not
    the default because its native PDF stack can crash worker processes.
    """
    mode = settings.PDF_EXTRACTION_MODE
    fast_result: List[Dict[str, Any]] = []

    if mode in {"auto", "fast"}:
        try:
            fast_result = extract_pdf_with_pymupdf(filepath, progress_callback=progress_callback)
            if fast_result and (mode == "fast" or not _pdf_requires_quality_extraction(filepath, fast_result)):
                return _merge_ocr_for_sparse_pages(filepath, fast_result, progress_callback)
        except Exception as exc:
            logger.warning("Fast PyMuPDF extraction failed, falling back: %s", exc)

    if mode in {"auto", "quality"}:
        try:
            result = extract_pdf_with_docling(filepath, progress_callback=progress_callback)
            if result:
                return _merge_ocr_for_sparse_pages(filepath, result, progress_callback)
        except Exception as exc:
            logger.warning("Docling extraction failed, falling back: %s", exc)

    if settings.PDF_USE_UNSTRUCTURED:
        try:
            result = extract_pdf_with_unstructured(filepath)
            if result:
                return _merge_ocr_for_sparse_pages(filepath, result, progress_callback)
        except Exception as e:
            # Unstructured may be installed but require native deps (poppler/pdfinfo).
            logger.warning(f"Unstructured extraction failed, falling back: {e}")
    
    try:
        result = extract_pdf_with_tables(filepath)
        if result:
            return _merge_ocr_for_sparse_pages(filepath, result, progress_callback)
    except Exception as e2:
        logger.warning(f"pdfplumber extraction failed, falling back: {e2}")
    
    try:
        result = fast_result or extract_pdf_with_pymupdf(filepath, progress_callback=progress_callback)
        if result:
            return _merge_ocr_for_sparse_pages(filepath, result, progress_callback)
    except Exception as e3:
        logger.warning(f"PyMuPDF extraction failed, falling back to OCR: {e3}")
    
    # Last resort: try OCR for image-based PDFs
    try:
        result = extract_pdf_with_ocr(filepath, progress_callback=progress_callback)
        if result:
            logger.info(f"Successfully extracted text from {filepath} using OCR")
            return result
    except Exception as e4:
        logger.warning(f"OCR extraction failed: {e4}")
    
    # If all extraction methods fail, return empty list
    logger.error(f"Could not extract text from {filepath} using any method")
    return []


def extract_pdf_with_docling(
    filepath: str,
    progress_callback: Optional[ProgressCallback] = None,
) -> List[Dict[str, Any]]:
    """Extract layout elements with Docling while retaining page provenance."""
    try:
        from docling.document_converter import DocumentConverter
    except Exception as exc:
        raise ImportError("docling not available") from exc

    _notify_progress(progress_callback, "extracting_layout")
    conversion = DocumentConverter().convert(filepath)
    document = conversion.document
    pages: List[Dict[str, Any]] = []
    table_index = 0

    for item, _level in document.iterate_items():
        provenance = list(getattr(item, "prov", None) or [])
        page = int(getattr(provenance[0], "page_no", 1) or 1) if provenance else 1
        label = str(getattr(item, "label", "text")).lower()
        chunk_type = "table" if "table" in label else "text"
        text = str(getattr(item, "text", "") or "").strip()
        if chunk_type == "table" and hasattr(item, "export_to_markdown"):
            try:
                text = str(item.export_to_markdown(document)).strip()
            except Exception:
                pass
        if not text:
            continue

        payload: Dict[str, Any] = {
            "text": text,
            "page": page,
            "chunk_type": chunk_type,
            "docling_label": label,
        }
        if chunk_type == "table":
            payload["table_index"] = table_index
            table_index += 1
        if provenance:
            bbox = getattr(provenance[0], "bbox", None)
            if bbox is not None:
                coords = [
                    getattr(bbox, key, None)
                    for key in ("l", "t", "r", "b")
                ]
                if all(value is not None for value in coords):
                    payload["bbox"] = json.dumps(coords)
        pages.append(payload)
    page_total = max((int(item.get("page") or 1) for item in pages), default=1)
    _notify_progress(progress_callback, "extracting", page_total, page_total)
    return pages


def extract_pdf_with_pymupdf(
    filepath: str,
    progress_callback: Optional[ProgressCallback] = None,
) -> List[Dict[str, Any]]:
    """Fallback PDF extraction with page numbers using PyMuPDF."""
    doc = fitz.open(filepath)
    pages = []

    total_pages = len(doc)
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({
                "text": text,
                "page": page_num + 1,
                "chunk_type": "text",
            })
        _notify_progress(progress_callback, "extracting", page_num + 1, total_pages)

    doc.close()
    return pages


def extract_pdf_with_unstructured(filepath: str) -> List[Dict[str, Any]]:
    """Use Unstructured to partition PDF into elements and extract tables.

    This function will raise ImportError when Unstructured isn't installed so
    callers can fall back to other extractors.
    """
    try:
        from unstructured.partition.pdf import partition_pdf
        from unstructured.documents.elements import Table
    except Exception as e:
        raise ImportError("unstructured not available") from e

    elements = partition_pdf(filename=filepath)
    pages: List[Dict[str, Any]] = []
    table_idx = 0

    for elem in elements:
        # Determine element type and page number
        elem_type = getattr(elem, "element_type", None) or elem.__class__.__name__
        page_num = None
        if hasattr(elem, "page_number"):
            page_num = getattr(elem, "page_number")
        elif getattr(elem, "metadata", None):
            page_num = elem.metadata.get("page_number") or elem.metadata.get("page")
        page_num = int(page_num) if page_num else 1

        if isinstance(elem, Table) or (isinstance(elem_type, str) and elem_type.lower() == "table"):
            rows = []
            for raw_row in getattr(elem, "rows", []) or []:
                row = []
                for cell in raw_row:
                    # Cells may be elements or lists of elements
                    if isinstance(cell, (list, tuple)):
                        cell_text = " ".join(getattr(c, "text", str(c)) for c in cell)
                    else:
                        cell_text = getattr(cell, "text", str(cell))
                    row.append(cell_text)
                rows.append(row)

            table_text = _table_to_markdown(rows)
            if table_text.strip():
                pages.append({
                    "text": table_text,
                    "page": page_num,
                    "chunk_type": "table",
                    "table_index": table_idx,
                })
                table_idx += 1
        else:
            text = getattr(elem, "text", str(elem) if elem else "")
            if text and text.strip():
                pages.append({
                    "text": text,
                    "page": page_num,
                    "chunk_type": "text",
                })

    return pages


def extract_pdf_with_tables(filepath: str) -> List[Dict[str, Any]]:
    """Detect tables with pdfplumber, remove table text from paragraphs, and keep table bboxes."""
    import pdfplumber

    pages: List[Dict[str, Any]] = []

    with pdfplumber.open(filepath) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.find_tables()
            table_bboxes = [table.bbox for table in tables]

            words = page.extract_words() or []
            paragraph_words = [
                word for word in words
                if not any(_is_word_inside_bbox(word, bbox) for bbox in table_bboxes)
            ]
            paragraph_text = _words_to_text(paragraph_words)

            if paragraph_text.strip():
                pages.append({
                    "text": paragraph_text,
                    "page": page_num,
                    "chunk_type": "text",
                })

            for table_index, table in enumerate(tables):
                table_text = _table_to_markdown(table.extract() or [])
                if table_text.strip():
                    # Normalize table bbox: [x0/W, y0/H, x1/W, y1/H]
                    W, H = float(page.width), float(page.height)
                    normalized_bbox = [
                        round(float(table.bbox[0]) / W, 4),
                        round(float(table.bbox[1]) / H, 4),
                        round(float(table.bbox[2]) / W, 4),
                        round(float(table.bbox[3]) / H, 4),
                    ]
                    pages.append({
                        "text": table_text,
                        "page": page_num,
                        "chunk_type": "table",
                        "bbox": json.dumps(normalized_bbox),
                        "table_index": table_index,
                    })

    return pages


def extract_pdf_with_ocr(
    filepath: str,
    page_numbers: List[int] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> List[Dict[str, Any]]:
    """Extract text from image-based PDFs using OCR (Tesseract via pdf2image).
    
    This function converts PDF pages to images and applies optical character recognition
    to extract text. Useful for scanned documents or PDFs where other extraction methods
    fail to retrieve sufficient text.
    
    Args:
        filepath: Path to the PDF file.
    
    Returns:
        List of dicts with keys: 'text', 'page', 'chunk_type' (always 'text').
        Returns empty list if pdf2image or pytesseract are unavailable.
    """
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError as e:
        logger.warning(f"OCR dependencies not available (pdf2image/pytesseract): {e}")
        return []

    if not shutil.which("tesseract"):
        logger.warning("OCR skipped because the tesseract binary is not installed or not on PATH")
        return []
    
    pages: List[Dict[str, Any]] = []
    
    try:
        selected_pages = sorted(set(page_numbers or [])) or None
        convert_kwargs = {
            "dpi": int(os.getenv("OCR_DPI", "300")),
            "fmt": "ppm",
            "thread_count": int(os.getenv("OCR_THREAD_COUNT", "2")),
        }
        ocr_languages = _resolve_tesseract_languages(
            pytesseract,
            os.getenv("OCR_LANGUAGES", "spa+eng"),
        )
        tesseract_config = os.getenv("OCR_TESSERACT_CONFIG", "")

        if selected_pages:
            page_iterable = selected_pages
        else:
            doc = None
            try:
                doc = fitz.open(filepath)
                page_iterable = range(1, len(doc) + 1)
            except Exception:
                page_iterable = [None]
            finally:
                if doc:
                    doc.close()

        page_iterable = list(page_iterable)
        total_requested = len(page_iterable)
        for request_index, requested_page in enumerate(page_iterable, start=1):
            try:
                page_convert_kwargs = dict(convert_kwargs)
                if requested_page is not None:
                    page_convert_kwargs["first_page"] = requested_page
                    page_convert_kwargs["last_page"] = requested_page

                images = convert_from_path(filepath, **page_convert_kwargs)
                for image_offset, image in enumerate(images):
                    page_num = requested_page or image_offset + 1
                    # Apply OCR to extract text from image
                    text = pytesseract.image_to_string(
                        image,
                        lang=ocr_languages,
                        config=tesseract_config,
                    )
                    text = text.strip()
                    
                    if text:
                        pages.append({
                            "text": text,
                            "page": page_num,
                            "chunk_type": "text",
                            "ocr_source": True,  # Flag to indicate this came from OCR
                        })
                    else:
                        logger.debug(f"OCR returned empty text for page {page_num}")
            except Exception as e:
                logger.warning(f"OCR failed for page {requested_page or 'unknown'}: {e}")
            finally:
                _notify_progress(progress_callback, "extracting_ocr", request_index, total_requested)
    except Exception as e:
        logger.warning(f"PDF to image conversion failed: {e}")
        return []
    
    return pages


def extract_pdf_images(
    doc_or_path: Any,
    min_width: int = 50,
    min_height: int = 50,
    min_size: int = 10240,
) -> Any:
    """Generator to yield extracted images from a PDF page-by-page.

    Accepts either a file path (str) or an open fitz.Document object.
    Yields dict: {"image_bytes": b"...", "page": int, "width": int, "height": int}
    """
    if not doc_or_path:
        return

    is_path = isinstance(doc_or_path, str)
    doc = None
    if is_path:
        try:
            doc = fitz.open(doc_or_path)
        except Exception as e:
            logger.warning(f"Could not open PDF with fitz for image extraction: {e}")
            return
    else:
        doc = doc_or_path

    try:
        for page_num, page in enumerate(doc):
            processed_xrefs = set()
            for img in page.get_images(full=True):
                xref = img[0]
                if xref in processed_xrefs:
                    continue
                processed_xrefs.add(xref)

                try:
                    pix = fitz.Pixmap(doc, xref)
                    width, height = pix.width, pix.height
                    
                    # Convert to RGB if it's CMYK or has alpha
                    if pix.n >= 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    img_bytes = pix.tobytes("png")
                    
                    # Skip tiny/decorative images (bullet points, spacers, logos)
                    if width < min_width or height < min_height or len(img_bytes) < min_size:
                        del img_bytes
                        del pix
                        continue

                    yield {
                        "image_bytes": img_bytes,
                        "page": page_num + 1,
                        "width": width,
                        "height": height,
                    }
                except Exception:
                    continue
    finally:
        if is_path and doc:
            doc.close()


def extract_docx(filepath: str) -> List[Dict[str, Any]]:
    """Extract text from DOCX files."""
    doc = docx.Document(filepath)
    full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

    return [{"text": full_text, "page": 1}] if full_text else []


def extract_txt(filepath: str) -> List[Dict[str, Any]]:
    """Extract text from TXT/Markdown files, preserving tables as separate chunks."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    if not text.strip():
        return []

    chunks = []
    current_text_lines = []
    table_lines = []
    in_table = False

    for line in text.splitlines():
        is_table_line = line.strip().startswith("|")

        if is_table_line:
            if not in_table:
                # flush any accumulated text first
                if current_text_lines:
                    chunk_text = "\n".join(current_text_lines).strip()
                    if chunk_text:
                        chunks.append({"text": chunk_text, "page": 1, "chunk_type": "text"})
                    current_text_lines = []
                in_table = True
            table_lines.append(line)
        else:
            if in_table:
                # flush the table
                table_text = "\n".join(table_lines).strip()
                if table_text:
                    chunks.append({"text": table_text, "page": 1, "chunk_type": "table"})
                table_lines = []
                in_table = False
            current_text_lines.append(line)

    # flush whatever's left
    if in_table and table_lines:
        chunks.append({"text": "\n".join(table_lines).strip(), "page": 1, "chunk_type": "table"})
    elif current_text_lines:
        chunk_text = "\n".join(current_text_lines).strip()
        if chunk_text:
            chunks.append({"text": chunk_text, "page": 1, "chunk_type": "text"})

    return chunks

# Change the chunk_document function input to take a file path and optional chunk size and overlap parameters. 
def chunk_document(
    filepath: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> List[Dict[str, Any]]:
    """
    Load a document, extract text per page, and split into semantic chunks.
    Accepts a file path and optional chunk size and overlap parameters. 
    If chunk size and overlap are not provided, defaults from settings will be used.
    Returns list of dicts with 'text', 'page', and 'chunk_index'.
    """

    ext = filepath.rsplit(".", 1)[-1].lower()

    # ── Extract text by file type ────────────────────
    if ext == "pdf":
        pages = (
            extract_pdf(filepath, progress_callback=progress_callback)
            if progress_callback
            else extract_pdf(filepath)
        )
    elif ext == "docx":
        pages = extract_docx(filepath)
    elif ext in ("txt", "md") or ext in CODE_EXTENSIONS:
        pages = extract_txt(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    pdf_doc = None
    if ext == "pdf":
        try:
            pdf_doc = fitz.open(filepath)
        except Exception as e:
            logger.warning(f"Could not open PDF with fitz: {e}")

    if not pages:
        if not pdf_doc or len(pdf_doc) == 0:
            if pdf_doc:
                pdf_doc.close()
            return []
    
    # Set chunk size and chunk overlap with defaults if not provided
    if not chunk_size:
        chunk_size = settings.CHUNK_SIZE
    if not chunk_overlap:
        chunk_overlap = settings.CHUNK_OVERLAP

    # ── LangChain recursive splitter ─────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, # Allow custom chunk size to be passed in for embedding
        chunk_overlap=chunk_overlap, # Allow custom chunk overlap to be passed in for embedding
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=_token_length,
    )

    all_chunks = []
    chunk_index = 0

    try:
        # Determine total pages to process
        max_page_in_data = max(page_data["page"] for page_data in pages) if pages else 0
        total_pages = max(max_page_in_data, len(pdf_doc) if pdf_doc else 0)

        # Set up image generator iterator
        image_iter = None
        next_image = None
        if pdf_doc:
            try:
                # pass pdf_doc to avoid opening it twice
                image_iter = iter(extract_pdf_images(pdf_doc))
                next_image = next(image_iter)
            except StopIteration:
                next_image = None
            except Exception as e:
                logger.warning(f"Could not initialize image iterator: {e}")

        # Group pages by page number for sequential access
        pages_by_num = {}
        for p_data in pages:
            pages_by_num.setdefault(p_data["page"], []).append(p_data)

        for page_num in range(1, total_pages + 1):
            # 1. Process text/table chunks for this page
            page_data_list = pages_by_num.get(page_num, [])
            for page_data in page_data_list:
                text = page_data["text"]
                chunk_type = page_data.get("chunk_type", "text")

                if chunk_type == "table":
                    all_chunks.append({
                        "text": text.strip(),
                        "page": page_num,
                        "chunk_index": chunk_index,
                        "chunk_type": "table",
                        "bbox": page_data.get("bbox", ""),
                        "table_index": page_data.get("table_index", 0),
                    })
                    chunk_index += 1
                    continue

                # Split this page's text
                splits = splitter.split_text(text)

                for split_text in splits:
                    if split_text.strip():
                        chunk = {
                            "text": split_text.strip(),
                            "page": page_num,
                            "chunk_index": chunk_index,
                            "chunk_type": chunk_type,
                        }

                        # Extract bbox for PDF text chunks
                        if pdf_doc and page_num <= len(pdf_doc):
                            try:
                                page_obj = pdf_doc[page_num - 1]
                                rects = page_obj.search_for(split_text.strip())
                                if rects:
                                    W, H = float(page_obj.rect.width), float(page_obj.rect.height)
                                    norm_rects = [
                                        [
                                            round(r.x0 / W, 4),
                                            round(r.y0 / H, 4),
                                            round(r.x1 / W, 4),
                                            round(r.y1 / H, 4)
                                        ]
                                        for r in rects
                                    ]
                                    chunk["bbox"] = json.dumps(norm_rects)
                            except Exception as e:
                                logger.warning(f"Bbox extraction error on page {page_num}: {e}")

                        all_chunks.append(chunk)
                        chunk_index += 1

            # 2. Attach any images that belong to this page (generating captions on-the-fly and discarding bytes)
            while next_image and next_image["page"] == page_num:
                img_bytes = next_image["image_bytes"]
                try:
                    # Generate caption immediately
                    from app.rag.vision import caption_image
                    caption = caption_image(img_bytes, page=page_num)

                    if caption:
                        all_chunks.append({
                            "text": caption,
                            "page": page_num,
                            "chunk_index": chunk_index,
                            "chunk_type": "text",
                            "is_image": True,
                            "image_caption": caption,
                        })
                        chunk_index += 1
                except Exception as e:
                    logger.warning(f"Failed to generate caption for image on page {page_num}: {e}")
                    fallback_text = f"Image on page {page_num}."
                    all_chunks.append({
                        "text": fallback_text,
                        "page": page_num,
                        "chunk_index": chunk_index,
                        "chunk_type": "text",
                        "is_image": True,
                        "image_caption": fallback_text,
                    })
                    chunk_index += 1
                finally:
                    # Explicitly remove reference to raw image bytes to free memory
                    if next_image:
                        next_image["image_bytes"] = None
                    if "img_bytes" in locals():
                        del img_bytes

                    # Fetch next image
                    try:
                        next_image = next(image_iter)
                    except StopIteration:
                        next_image = None
                    except Exception as e:
                        logger.warning(f"Error getting next image from iterator: {e}")
                        next_image = None

            # Collect garbage at the end of each page to prevent memory buildup
            import gc
            gc.collect()
            _notify_progress(progress_callback, "chunking", page_num, total_pages)

    finally:
        if pdf_doc:
            pdf_doc.close()

    return _annotate_hierarchy(all_chunks)


def get_page_count(filepath: str) -> int:
    """Get total page count of a document."""
    ext = filepath.rsplit(".", 1)[-1].lower()

    if ext == "pdf":
        doc = fitz.open(filepath)
        count = len(doc)
        doc.close()
        return count

    return 1  # DOCX, TXT, MD are treated as single-page
