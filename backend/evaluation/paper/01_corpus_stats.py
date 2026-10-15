"""§3.10 Corpus — cifras descriptivas del corpus indexado.

Número de documentos, páginas, fragmentos, idioma, dominio y proporción de
páginas que requirieron OCR. El criterio de OCR es el mismo que aplica la
ingesta (``chunker._page_has_enough_text``: < 40 caracteres no blancos en la
capa de texto del PDF).

    python evaluation/paper/01_corpus_stats.py --label A-mac --user-email tu@correo
    python evaluation/paper/01_corpus_stats.py --label A-mac --manifest evaluation/paper/corpus_manifest.csv

El manifiesto (opcional) es un CSV con columnas ``filename,domain,language``
donde ``filename`` es el nombre original del archivo subido. Si se omite el
idioma se estima con una heurística de palabras vacías es/en.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import _common

ES_STOPWORDS = {"el", "la", "los", "las", "de", "del", "que", "y", "en", "un", "una", "por", "con", "para",
                "se", "su", "al", "es", "lo", "como", "más", "pero", "sus", "este", "esta", "entre"}
EN_STOPWORDS = {"the", "of", "and", "to", "in", "a", "is", "that", "for", "on", "with", "as", "by", "are",
                "this", "be", "from", "or", "an", "which", "was", "were", "it", "at", "these"}


def guess_language(text: str) -> str:
    words = re.findall(r"[a-záéíóúñü]+", text.casefold())
    es = sum(word in ES_STOPWORDS for word in words)
    en = sum(word in EN_STOPWORDS for word in words)
    if es + en < 20:
        return "desconocido"
    return "es" if es > en else "en"


def inspect_pdf(path: Path, min_chars: int) -> Dict[str, Any]:
    import pymupdf

    sample: List[str] = []
    sparse_pages = 0
    with pymupdf.open(path) as pdf:
        pages = pdf.page_count
        for page in pdf:
            text = page.get_text("text") or ""
            if len(re.sub(r"\s+", "", text)) < min_chars:
                sparse_pages += 1
            if sum(map(len, sample)) < 30000:
                sample.append(text)
    return {"pages_pdf": pages, "ocr_pages": sparse_pages, "text_sample": "\n".join(sample)}


def load_manifest(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if not path:
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {_common.normalize_text(row["filename"]): row for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True)
    _common.add_user_args(parser)
    parser.add_argument("--manifest", type=Path, help="CSV filename,domain,language (opcional).")
    args = parser.parse_args()

    _common.bootstrap()
    from app.config import get_settings
    from app.database import SessionLocal
    from app.models import Document
    from app.rag.chunker import OCR_MIN_TEXT_CHARS_PER_PAGE

    settings = get_settings()
    user_id = _common.resolve_user_id(args.user_email, args.user_id)
    manifest = load_manifest(args.manifest)

    db = SessionLocal()
    try:
        documents = (
            db.query(Document)
            .filter(Document.user_id == user_id, Document.is_deleted.is_(False))
            .order_by(Document.original_name)
            .all()
        )
        rows: List[Dict[str, Any]] = []
        for document in documents:
            row: Dict[str, Any] = {
                "filename": document.original_name,
                "status": document.status,
                "extension": Path(document.original_name).suffix.lower().lstrip("."),
                "pages": document.page_count or 0,
                "chunks": document.chunk_count or 0,
                "ocr_pages": None,
            }
            stored = Path(settings.UPLOAD_DIR) / str(user_id) / document.filename
            text_sample = ""
            if row["extension"] == "pdf" and stored.exists():
                try:
                    inspected = inspect_pdf(stored, OCR_MIN_TEXT_CHARS_PER_PAGE)
                    row["ocr_pages"] = inspected["ocr_pages"]
                    row["pages"] = row["pages"] or inspected["pages_pdf"]
                    text_sample = inspected["text_sample"]
                except Exception as exc:
                    row["inspect_error"] = str(exc)
            elif stored.exists() and row["extension"] in {"txt", "md"}:
                text_sample = stored.read_text(encoding="utf-8", errors="ignore")[:30000]
            elif stored.exists() and row["extension"] == "docx":
                try:
                    import docx

                    text_sample = "\n".join(p.text for p in docx.Document(str(stored)).paragraphs)[:30000]
                except Exception as exc:
                    row["inspect_error"] = str(exc)
            entry = manifest.get(_common.normalize_text(document.original_name), {})
            row["domain"] = entry.get("domain") or "sin asignar"
            row["language"] = entry.get("language") or (guess_language(text_sample) if text_sample else "desconocido")
            row["language_source"] = "manifiesto" if entry.get("language") else "heurística"
            rows.append(row)
    finally:
        db.close()

    ready = [row for row in rows if row["status"] == "ready"]
    pdfs = [row for row in ready if row["ocr_pages"] is not None]
    total_pages = sum(row["pages"] for row in ready)
    pdf_pages = sum(row["pages"] for row in pdfs)
    ocr_pages = sum(row["ocr_pages"] for row in pdfs)
    summary = {
        "label": args.label,
        "documents_total": len(rows),
        "documents_ready": len(ready),
        "documents_not_ready": Counter(row["status"] for row in rows if row["status"] != "ready"),
        "by_extension": Counter(row["extension"] for row in ready),
        "pages_total": total_pages,
        "chunks_total": sum(row["chunks"] for row in ready),
        "chunks_per_document_median": _common.percentile([row["chunks"] for row in ready], 50),
        "pdf_pages_inspected": pdf_pages,
        "ocr_pages": ocr_pages,
        "ocr_pages_pct": round(100 * ocr_pages / pdf_pages, 2) if pdf_pages else None,
        "documents_with_any_ocr_page": sum(1 for row in pdfs if row["ocr_pages"]),
        "by_language": Counter(row["language"] for row in ready),
        "by_domain": Counter(row["domain"] for row in ready),
        "domains_count": len({row["domain"] for row in ready if row["domain"] != "sin asignar"}),
    }

    out_dir = _common.results_dir(args.label)
    _common.write_json(out_dir / "corpus.json", {"summary": summary, "documents": rows})
    with (out_dir / "corpus_documents.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["filename", "status", "extension", "pages", "chunks", "ocr_pages", "domain", "language", "language_source"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    def counter_text(counter: Counter) -> str:
        return ", ".join(f"{key}: {value}" for key, value in counter.most_common()) or "—"

    markdown = "\n".join([
        f"### Corpus — {args.label}",
        "",
        "| Medida | Valor |",
        "|---|---|",
        f"| Documentos indexados (listos) | {summary['documents_ready']} de {summary['documents_total']} |",
        f"| Por formato | {counter_text(summary['by_extension'])} |",
        f"| Páginas | {total_pages} |",
        f"| Fragmentos hijo indexados | {summary['chunks_total']} (mediana por documento: {_common.fmt(summary['chunks_per_document_median'], 0)}) |",
        f"| Páginas que requirieron OCR | {ocr_pages} de {pdf_pages} ({_common.fmt(summary['ocr_pages_pct'], 2)} %) en {summary['documents_with_any_ocr_page']} documentos |",
        f"| Idioma | {counter_text(summary['by_language'])} |",
        f"| Dominio | {counter_text(summary['by_domain'])} |",
        "",
        "Idioma por heurística de palabras vacías salvo que el manifiesto lo indique; revísalo en corpus_documents.csv.",
    ]) + "\n"
    (out_dir / "corpus.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Guardado en {out_dir}")


if __name__ == "__main__":
    main()
