"""RAG query pipeline: retrieve relevant chunks, generate answer with source citations."""

from __future__ import annotations

from pathlib import Path


def load_vectorstore(persist_dir: str | Path = "chroma_db") -> object:
    """Load an existing ChromaDB vectorstore from disk.

    Args:
        persist_dir: Directory where ChromaDB was persisted.

    Returns:
        ChromaDB vectorstore instance.
    """
    raise NotImplementedError("Phase 2")


def build_rag_chain(vectorstore: object, model: str = "llama3.2", k: int = 4) -> object:
    """Build a LangChain RAG chain using the given vectorstore.

    Args:
        vectorstore: ChromaDB vectorstore with embedded course documents.
        model: Ollama model name to use for generation.
        k: Number of chunks to retrieve per query.

    Returns:
        LangChain runnable chain (invoke with ``{"question": "..."}``)
    """
    raise NotImplementedError("Phase 2")


def ask(chain: object, question: str) -> dict:
    """Run a question through the RAG chain and return answer + sources.

    Args:
        chain: LangChain RAG chain from :func:`build_rag_chain`.
        question: Natural-language question.

    Returns:
        Dict with keys ``answer`` (str) and ``sources`` (list of chunk metadata dicts).
    """
    raise NotImplementedError("Phase 2")
