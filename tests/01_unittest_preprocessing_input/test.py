from pathlib import Path
import sys

import fitz

ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from test_support import load_module

PDF_PATHS = (
    ROOT / "history-of-russia-part-2.pdf",
    ROOT / "russian-history-part-i.pdf",
)


def test_create_directory_loader_for_pdfs():
    module = load_module("preprocessing")
    preprocessor = module.Preprocessor(ROOT, {".pdf": module.PyMuPDFLoader}, 500, 50)

    loader = preprocessor.create_directory_loader(".pdf")
    documents = loader.load()

    expected_pages = 0
    for pdf_path in PDF_PATHS:
        pdf = fitz.open(pdf_path)
        expected_pages += pdf.page_count
        pdf.close()

    assert isinstance(loader, module.DirectoryLoader)
    assert len(documents) == expected_pages


def test_preprocess_data_returns_non_empty_chunks_for_real_pdfs():
    module = load_module("preprocessing")
    preprocessor = module.Preprocessor(ROOT, {".pdf": module.PyMuPDFLoader}, 500, 50)

    documents = preprocessor.create_directory_loader(".pdf").load()
    chunks = preprocessor.preprocess_data()

    assert chunks
    assert len(chunks) >= len(documents)
