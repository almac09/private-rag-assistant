"""Document ingestion pipeline: load PDFs, chunk, embed, store in ChromaDB."""

from __future__ import annotations

from pathlib import Path


def load_pdfs(source_dir: str | Path) -> list:
    """Load all PDFs from *source_dir* using LangChain's PyPDFLoader.

    Args:
        source_dir: Directory containing PDF files.

    Returns:
        List of LangChain Document objects, one per page.
    """
    raise NotImplementedError("Phase 1 — implement in notebooks/01_ingest.ipynb first")


def chunk_documents(documents: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """Split documents into overlapping text chunks.

    Args:
        documents: List of LangChain Document objects.
        chunk_size: Target characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of chunked Document objects.
    """
    raise NotImplementedError("Phase 1")


def build_vectorstore(chunks: list, persist_dir: str | Path = "chroma_db") -> object:
    """Embed chunks with Ollama nomic-embed-text and persist to ChromaDB.

    Args:
        chunks: Chunked Document objects from :func:`chunk_documents`.
        persist_dir: Directory for ChromaDB persistence.

    Returns:
        ChromaDB vectorstore instance.
    """
    raise NotImplementedError("Phase 1")
