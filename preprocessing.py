from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pymupdf as fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _document_sort_key(document: Document) -> tuple[str, int, int]:
    metadata = document.metadata or {}
    source = str(metadata.get("source", ""))
    page = int(metadata.get("page", -1))
    start_index = int(metadata.get("start_index", -1))
    return source, page, start_index


class PyMuPDFLoader:
    """Minimal PDF loader."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)

    def load(self) -> list[Document]:
        pdf = fitz.open(self.file_path)
        documents = []
        for page_index in range(pdf.page_count):
            page = pdf.load_page(page_index)
            documents.append(
                Document(
                    page_content=page.get_text("text"),
                    metadata={"source": str(self.file_path), "page": page_index},
                )
            )
        pdf.close()
        return documents


class DirectoryLoader:
    """Directory loader for local PDF files."""

    def __init__(
        self,
        path: str | Path,
        glob: str,
        loader_cls: Any,
        silent_errors: bool = False,
    ) -> None:
        self.path = Path(path)
        self.glob = glob
        self.loader_cls = loader_cls
        self.silent_errors = silent_errors

    def load(self) -> list[Document]:
        documents = []
        for file_path in sorted(self.path.glob(self.glob)):
            if not file_path.is_file():
                continue
            try:
                documents.extend(self.loader_cls(file_path).load())
            except Exception:
                if not self.silent_errors:
                    raise
        return documents


class Preprocessor:
    """Load local documents and split them into deterministic chunks."""

    def __init__(
        self,
        path2data: str | Path,
        loaders: Mapping[str, Any],
        chunk_length: int,
        chunk_overlap: int,
    ) -> None:
        self.path2data = Path(path2data)
        self.loaders = dict(loaders)
        self.chunk_length = chunk_length
        self.chunk_overlap = chunk_overlap

    def create_directory_loader(self, file_type: str) -> DirectoryLoader:
        """Create a directory loader for one file extension."""
        if file_type not in self.loaders:
            raise ValueError(f"Unsupported file type: {file_type}")
        return DirectoryLoader(
            path=self.path2data,
            glob=f"*{file_type}",
            loader_cls=self.loaders[file_type],
            silent_errors=False,
        )

    def preprocess_data(self) -> list[Document]:
        """Load the documents and split them into normalized chunks."""
        loader = self.create_directory_loader(".pdf")
        docs = loader.load()

        if not docs:
            raise FileNotFoundError("No PDF files found in the data directory.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_length,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = text_splitter.split_documents(docs)

        processed = []
        for chunk in chunks:
            normalized = _normalize_text(chunk.page_content)
            if not normalized:
                continue
            chunk.page_content = normalized
            processed.append(chunk)

        processed.sort(key=_document_sort_key)

        for idx, chunk in enumerate(processed):
            chunk.metadata["chunk_index"] = idx

        return processed