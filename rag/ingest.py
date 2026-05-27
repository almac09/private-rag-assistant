"""
Document ingestion pipeline: load ECB speeches CSV, chunk, embed, and store in ChromaDB.
"""

import os
from pathlib import Path
from typing import Optional
import pandas as pd

import chromadb
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


def load_speeches_csv(csv_path: str) -> list:
    """
    Load ECB speeches from CSV and convert to LangChain Document objects.

    Args:
        csv_path: Path to ECB speeches CSV (columns: date, year, speakers, title,
                  subtitle, contents).

    Returns:
        List of LangChain Document objects with metadata.
    """
    df = pd.read_csv(csv_path)
    documents = []

    for idx, row in df.iterrows():
        # Use speech contents as page_content
        page_content = row.get("contents", "")
        if not page_content or pd.isna(page_content):
            continue

        # Attach all metadata
        metadata = {
            "date": str(row.get("date", "")),
            "year": str(row.get("year", "")),
            "speakers": str(row.get("speakers", "")),
            "title": str(row.get("title", "")),
            "subtitle": str(row.get("subtitle", "")),
            "source": csv_path,
            "row_index": idx,
        }

        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)

    return documents


def chunk_documents(
    documents: list,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list:
    """
    Split documents into chunks using recursive character splitter.

    Args:
        documents: List of LangChain Document objects.
        chunk_size: Characters per chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        List of chunked Document objects (metadata preserved).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def ingest_to_chromadb(
    csv_path: str,
    chroma_dir: str = "./chroma_db",
    ollama_model: str = "nomic-embed-text",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> Chroma:
    """
    End-to-end ingestion: load ECB speeches CSV → chunk → embed → store in ChromaDB.

    Args:
        csv_path: Path to ECB speeches CSV file.
        chroma_dir: Where to persist ChromaDB.
        ollama_model: Embedding model (must be running in Ollama).
        chunk_size: Characters per chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        Chroma vector store instance.

    Raises:
        FileNotFoundError: If csv_path doesn't exist.
        ValueError: If CSV is empty or malformed.
        Exception: If Ollama is not reachable.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(f"Loading ECB speeches from {csv_path}...")
    documents = load_speeches_csv(csv_path)
    if not documents:
        raise ValueError(f"No valid speeches found in {csv_path}")
    print(f"Loaded {len(documents)} speeches.")

    print("Chunking documents...")
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    print(f"Created {len(chunks)} chunks.")

    print(f"Initializing Ollama embeddings ({ollama_model})...")
    embeddings = OllamaEmbeddings(model=ollama_model)

    print("Embedding and storing in ChromaDB...")
    os.makedirs(chroma_dir, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_dir,
        collection_name="ecb_speeches",
    )
    vectorstore.persist()
    print(f"Ingestion complete. ChromaDB persisted to {chroma_dir}")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Collection: ecb_speeches")

    return vectorstore


def load_chromadb(
    chroma_dir: str = "./chroma_db",
    ollama_model: str = "nomic-embed-text",
) -> Chroma:
    """
    Load an existing ChromaDB vector store.

    Args:
        chroma_dir: Path to persisted ChromaDB.
        ollama_model: Embedding model (must match the one used during ingestion).

    Returns:
        Chroma vector store instance.
    """
    embeddings = OllamaEmbeddings(model=ollama_model)
    vectorstore = Chroma(
        persist_directory=chroma_dir,
        embedding_function=embeddings,
        collection_name="ecb_speeches",
    )
    return vectorstore
