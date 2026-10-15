"""Ingesta reproducible del corpus de evaluación en un usuario dedicado.

Crea (o reutiliza) un usuario de evaluación, copia cada archivo de la
carpeta del corpus a UPLOAD_DIR/<user_id>/ y ejecuta la misma función de
ingesta que usa la aplicación (``ingest_document``), de forma síncrona.
El usuario no tiene contraseña utilizable: solo existe para aislar el corpus
de evaluación de los documentos personales.

Escribe además ``corpus_files.csv`` con el SHA-256 de cada archivo, para
publicar el corpus como lista de DOI/URL + hash sin redistribuir los PDF y
para comprobar que ambas máquinas indexaron exactamente los mismos archivos.

    python evaluation/paper/01a_ingest_corpus.py --corpus ~/Downloads/ATLAS_corpus_evaluacion --label A-mac
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import secrets
import shutil
import time
import uuid
from pathlib import Path

import _common

DEFAULT_EMAIL = "atlas-eval@eval.test"
EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True, help="Carpeta con los documentos del corpus.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--user-email", default=DEFAULT_EMAIL)
    args = parser.parse_args()

    corpus = args.corpus.expanduser().resolve()
    files = sorted(p for p in corpus.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not files:
        raise SystemExit(f"No hay documentos en {corpus}")

    _common.bootstrap()
    from app.auth import hash_password
    from app.config import get_settings
    from app.database import SessionLocal
    from app.models import Document, User
    from app.services.document_ingestion import ingest_document

    settings = get_settings()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == args.user_email).first()
        if user is None:
            user = User(
                username=args.user_email.split("@")[0].replace("-", "_"),
                email=args.user_email,
                hashed_password=hash_password(secrets.token_urlsafe(32)),
                is_verified=True,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            print(f"Usuario de evaluación creado: {args.user_email} ({user.id})")
        user_id = str(user.id)
        existing = {
            d.original_name: d
            for d in db.query(Document).filter(Document.user_id == user.id, Document.is_deleted.is_(False))
        }
    finally:
        db.close()

    rows = []
    for index, path in enumerate(files, 1):
        digest = sha256(path)
        current = existing.get(path.name)
        if current is not None and current.status == "ready":
            print(f"  {index}/{len(files)} {path.name}: ya indexado")
            rows.append({"filename": path.name, "sha256": digest, "status": "ready", "seconds": ""})
            continue
        db = SessionLocal()
        try:
            if current is not None:
                document = db.get(Document, current.id)
                stored = Path(settings.UPLOAD_DIR) / user_id / document.filename
            else:
                stored_name = f"{uuid.uuid4().hex}{path.suffix.lower()}"
                stored = Path(settings.UPLOAD_DIR) / user_id / stored_name
                document = Document(
                    user_id=user_id, filename=stored_name, original_name=path.name,
                    file_size=path.stat().st_size, status="pending",
                )
                db.add(document)
                db.commit()
                db.refresh(document)
            stored.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, stored)
            document_id = str(document.id)
        finally:
            db.close()

        start = time.perf_counter()
        ingest_document(document_id=document_id, filepath=str(stored), original_name=path.name, user_id=user_id)
        elapsed = time.perf_counter() - start
        db = SessionLocal()
        try:
            document = db.get(Document, uuid.UUID(document_id))
            status, chunks, error = document.status, document.chunk_count, document.error_message
        finally:
            db.close()
        print(f"  {index}/{len(files)} {path.name}: {status}, {chunks} fragmentos, {elapsed:.0f}s"
              + (f" ERROR {error}" if error else ""))
        rows.append({"filename": path.name, "sha256": digest, "status": status, "seconds": round(elapsed, 1)})

    out = _common.results_dir(args.label) / "corpus_files.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "sha256", "status", "seconds"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Usuario: {args.user_email} ({user_id}). Manifiesto con SHA-256 en {out}")


if __name__ == "__main__":
    main()
