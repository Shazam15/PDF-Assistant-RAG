from pathlib import Path
import sys
import types

import pytest

from app.rag import chunker
from app.rag.chunker import _table_to_markdown, chunk_document, get_page_count


def test_txt_extraction_and_chunking(tmp_path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("This is a sample text file for chunking.", encoding="utf-8")

    chunks = chunk_document(str(file_path))

    assert len(chunks) >= 1
    assert chunks[0]["page"] == 1
    assert "sample text file" in chunks[0]["text"]


def test_empty_txt_returns_no_chunks(tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("   \n", encoding="utf-8")

    assert chunk_document(str(file_path)) == []


def test_unsupported_extension_raises_value_error(tmp_path):
    file_path = tmp_path / "data.csv"
    file_path.write_text("a,b,c", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        chunk_document(str(file_path))


def test_get_page_count_for_txt_returns_one(tmp_path):
    file_path = tmp_path / "single.txt"
    file_path.write_text("hello", encoding="utf-8")

    assert get_page_count(str(file_path)) == 1


def test_table_to_markdown_cleans_cells_and_escapes_pipes():
    rows = [
        ["Name", "Age", "Role"],
        [" Asha\nRao ", 24, "Admin | Owner"],
        [None, "  ", None],
        ["Ravi", 28],
    ]

    assert _table_to_markdown(rows) == "\n".join([
        "| Name | Age | Role |",
        "| --- | --- | --- |",
        "| Asha Rao | 24 | Admin \\| Owner |",
        "| Ravi | 28 |  |",
    ])


def test_pdf_table_detection_separates_table_from_paragraph(monkeypatch):
    class FakeTable:
        bbox = (40, 90, 300, 160)

        def extract(self):
            return [["Name", "Amount"], ["Alpha", "$10"]]

    class FakePage:
        width = 400
        height = 200

        def find_tables(self):
            return [FakeTable()]

        def extract_words(self):
            return [
                {"text": "Intro", "x0": 40,  "x1": 70, "top": 20, "bottom": 30},
                {"text": "paragraph", "x0":  75, "x1": 140, "top": 20, "bottom": 30},
                {"text": "Name", "x0": 45,  "x1": 80, "top": 100, "bottom": 110},
                {"text": "Amount", "x0": 160, "x1": 220, "top": 100, "bottom": 110},
                {"text": "Alpha", "x0": 45,  "x1": 85, "top": 125, "bottom": 135},
                {"text": "$10", "x0": 160,  "x1": 185, "top": 125, "bottom": 135},
            ]

    class FakePdf:
        pages = [FakePage()]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    fake_pdfplumber = types.SimpleNamespace(open=lambda _filepath: FakePdf())
    monkeypatch.setitem(sys.modules, "pdfplumber", fake_pdfplumber)
    monkeypatch.setattr(chunker, "extract_pdf_images", lambda _filepath: [])

    chunks = chunk_document("report.pdf")

    assert len(chunks) == 2
    assert chunks[0]["chunk_type"] == "text"
    assert chunks[0]["text"] == "Intro paragraph"
    assert "Name" not in chunks[0]["text"]
    assert chunks[1]["chunk_type"] == "table"
    assert chunks[1]["bbox"] == "[0.1, 0.45, 0.75, 0.8]"
    assert "| Name | Amount |" in chunks[1]["text"]
    assert "| Alpha | $10 |" in chunks[1]["text"]


def test_unstructured_table_detection(monkeypatch):
    # Create fake Unstructured Table and Text element classes
    class FakeTableClass:
        pass

    class FakeTable(FakeTableClass):
        def __init__(self):
            self.rows = [["Name", "Amount"], ["Delta", "$40"]]
            self.page_number = 3

    class FakeText:
        def __init__(self):
            self.text = "Intro paragraph"
            self.page_number = 3

    def fake_partition_pdf(filename):
        return [FakeText(), FakeTable()]

    # Insert fake unstructured modules
    monkeypatch.setitem(sys.modules, "unstructured", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "unstructured.partition", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "unstructured.partition.pdf", types.SimpleNamespace(partition_pdf=fake_partition_pdf))
    monkeypatch.setitem(sys.modules, "unstructured.documents", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "unstructured.documents.elements", types.SimpleNamespace(Table=FakeTableClass))

    monkeypatch.setattr(chunker, "extract_pdf_images", lambda _filepath: [])

    chunks = chunk_document("sample.pdf")

    # Expect two chunks: text then table
    assert len(chunks) >= 2
    assert chunks[0]["chunk_type"] == "text"
    assert "Intro paragraph" in chunks[0]["text"]
    # find a table chunk
    table_chunks = [c for c in chunks if c.get("chunk_type") == "table"]
    assert table_chunks, "No table chunks produced by Unstructured path"
    assert table_chunks[0]["page"] == 3
    assert "| Name | Amount |" in table_chunks[0]["text"]
    assert "| Delta | $40 |" in table_chunks[0]["text"]


def test_pdf_image_captioning_on_the_fly(monkeypatch):
    # Mock extract_pdf_images to yield one image on page 1
    def fake_extract_images(doc_or_path, **kwargs):
        yield {
            "image_bytes": b"fake_png_bytes",
            "page": 1,
            "width": 100,
            "height": 100,
        }

    monkeypatch.setattr(chunker, "extract_pdf_images", fake_extract_images)

    # Mock caption_image to return a custom caption
    from app.rag import vision
    monkeypatch.setattr(vision, "caption_image", lambda img_bytes, page=None: f"Captured image on page {page}")

    # Mock extract_pdf to return a text page
    def fake_extract_pdf(filepath):
        return [{"text": "Hello world on page 1", "page": 1, "chunk_type": "text"}]

    monkeypatch.setattr(chunker, "extract_pdf", fake_extract_pdf)

    # Mock fitz.open
    class FakePage:
        rect = type('Rect', (), {'width': 100, 'height': 100})()
        def search_for(self, text):
            return []

    class FakePdf:
        def __init__(self, *args, **kwargs):
            pass
        def __len__(self):
            return 1
        def __getitem__(self, idx):
            return FakePage()
        def close(self):
            pass

    import fitz
    monkeypatch.setattr(fitz, "open", lambda *args, **kwargs: FakePdf())

    # Run chunk_document
    chunks = chunk_document("dummy.pdf")

    # The result should contain the text chunk and the image chunk
    assert len(chunks) == 2
    assert chunks[0]["text"] == "Hello world on page 1"
    assert chunks[0]["page"] == 1
    
    assert chunks[1]["text"] == "Captured image on page 1"
    assert chunks[1]["page"] == 1
    assert chunks[1]["is_image"] is True
    assert chunks[1]["image_caption"] == "Captured image on page 1"
    # Ensure image_bytes is not in the chunk dictionary (preventing memory leak)
    assert "image_bytes" not in chunks[1]


def test_ocr_extraction_with_mock_pdf2image_and_pytesseract(tmp_path, monkeypatch):
    """Test OCR extraction from scanned PDFs using mocked pdf2image and pytesseract."""
    from PIL import Image
    from io import BytesIO
    
    # Create a simple test image in memory
    def create_test_image():
        img = Image.new('RGB', (200, 100), color='white')
        return img
    
    # Mock pdf2image.convert_from_path to return image objects
    def mock_convert_from_path(filepath, dpi=200, fmt='ppm'):
        return [create_test_image(), create_test_image()]
    
    # Mock pytesseract.image_to_string to return text
    def mock_image_to_string(image, lang=None):
        return "This is text extracted from scanned document page"
    
    monkeypatch.setattr("pdf2image.convert_from_path", mock_convert_from_path)
    monkeypatch.setattr("pytesseract.image_to_string", mock_image_to_string)
    
    # Test the OCR extraction function
    from app.rag.chunker import extract_pdf_with_ocr
    
    # Create a dummy PDF file (it won't actually be read by the mock)
    test_pdf = tmp_path / "scanned.pdf"
    test_pdf.write_bytes(b"fake pdf content")
    
    result = extract_pdf_with_ocr(str(test_pdf))
    
    # Verify results
    assert len(result) == 2  # Two pages
    assert result[0]["page"] == 1
    assert result[1]["page"] == 2
    assert result[0]["chunk_type"] == "text"
    assert result[1]["chunk_type"] == "text"
    assert "scanned document" in result[0]["text"]
    assert "scanned document" in result[1]["text"]
    assert result[0].get("ocr_source") is True
    assert result[1].get("ocr_source") is True


def test_ocr_fallback_when_other_methods_fail(tmp_path, monkeypatch):
    """Test that OCR is called as fallback when other extraction methods return empty."""
    from PIL import Image
    
    def create_test_image():
        img = Image.new('RGB', (200, 100), color='white')
        return img
    
    # Mock extract_pdf_with_unstructured to return empty
    def mock_unstructured_extract(filepath):
        raise ImportError("Unstructured not available")
    
    # Mock extract_pdf_with_tables to return empty
    def mock_tables_extract(filepath):
        return []
    
    # Mock extract_pdf_with_pymupdf to return empty
    def mock_pymupdf_extract(filepath):
        return []
    
    # Mock pdf2image and pytesseract to succeed
    def mock_convert_from_path(filepath, dpi=200, fmt='ppm'):
        return [create_test_image()]
    
    def mock_image_to_string(image, lang=None):
        return "Extracted text via OCR fallback"
    
    # Apply monkeypatches for the extraction methods
    monkeypatch.setattr("app.rag.chunker.extract_pdf_with_unstructured", mock_unstructured_extract)
    monkeypatch.setattr("app.rag.chunker.extract_pdf_with_tables", mock_tables_extract)
    monkeypatch.setattr("app.rag.chunker.extract_pdf_with_pymupdf", mock_pymupdf_extract)
    monkeypatch.setattr("pdf2image.convert_from_path", mock_convert_from_path)
    monkeypatch.setattr("pytesseract.image_to_string", mock_image_to_string)
    
    # Create a dummy PDF file
    test_pdf = tmp_path / "image_only.pdf"
    test_pdf.write_bytes(b"fake pdf content")
    
    # Test extract_pdf function which should fallback to OCR
    result = chunker.extract_pdf(str(test_pdf))
    
    # Verify OCR was used as fallback
    assert len(result) == 1
    assert result[0]["page"] == 1
    assert "OCR fallback" in result[0]["text"]
    assert result[0].get("ocr_source") is True


def test_ocr_handles_missing_dependencies(tmp_path, monkeypatch):
    """Test that OCR gracefully handles missing pdf2image or pytesseract."""
    from app.rag.chunker import extract_pdf_with_ocr
    
    # Mock import to raise ImportError
    def mock_import_error(*args, **kwargs):
        raise ImportError("pdf2image not available")
    
    # Patch the import in the OCR function
    import sys
    # Create fake module that raises ImportError when imported
    monkeypatch.setitem(sys.modules, "pdf2image", None)
    monkeypatch.setitem(sys.modules, "pytesseract", None)
    
    # Create a dummy PDF file
    test_pdf = tmp_path / "missing_deps.pdf"
    test_pdf.write_bytes(b"fake pdf content")
    
    # Should return empty list when dependencies are missing
    result = extract_pdf_with_ocr(str(test_pdf))
    assert result == []


def test_ocr_handles_corrupted_pdf(tmp_path, monkeypatch):
    """Test that OCR handles corrupted PDFs gracefully."""
    from app.rag.chunker import extract_pdf_with_ocr
    
    # Mock pdf2image.convert_from_path to raise exception
    def mock_convert_error(filepath, dpi=200, fmt='ppm'):
        raise Exception("PDF is corrupted or unreadable")
    
    monkeypatch.setattr("pdf2image.convert_from_path", mock_convert_error)
    
    # Create a dummy PDF file
    test_pdf = tmp_path / "corrupted.pdf"
    test_pdf.write_bytes(b"corrupted pdf content")
    
    # Should handle the exception and return empty list
    result = extract_pdf_with_ocr(str(test_pdf))
    assert result == []


def test_ocr_skips_empty_results(tmp_path, monkeypatch):
    """Test that OCR skips pages where no text is extracted."""
    from PIL import Image
    
    def create_test_image():
        img = Image.new('RGB', (200, 100), color='white')
        return img
    
    # Mock to return 3 pages but only extract text from pages 1 and 3
    call_count = [0]
    def mock_image_to_string(image, lang=None):
        call_count[0] += 1
        if call_count[0] == 2:  # Skip page 2
            return ""
        return f"Text from page {call_count[0]}"
    
    def mock_convert_from_path(filepath, dpi=200, fmt='ppm'):
        return [create_test_image(), create_test_image(), create_test_image()]
    
    monkeypatch.setattr("pdf2image.convert_from_path", mock_convert_from_path)
    monkeypatch.setattr("pytesseract.image_to_string", mock_image_to_string)
    
    from app.rag.chunker import extract_pdf_with_ocr
    
    test_pdf = tmp_path / "partial.pdf"
    test_pdf.write_bytes(b"fake pdf")
    
    result = extract_pdf_with_ocr(str(test_pdf))
    
    # Should only return pages with extracted text
    assert len(result) == 2
    assert result[0]["page"] == 1
    assert result[1]["page"] == 3
    assert "Text from page 1" in result[0]["text"]
    assert "Text from page 3" in result[1]["text"]


