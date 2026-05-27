"""
Tests for document ingestion pipeline (ECB speeches CSV).
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from rag.ingest import load_speeches_csv, chunk_documents, ingest_to_chromadb


@pytest.fixture
def sample_csv(tmp_path):
    """Create a temporary CSV with sample ECB speech data."""
    csv_data = {
        "date": ["2023-01-15", "2023-02-20"],
        "year": [2023, 2023],
        "speakers": ["Christine Lagarde", "Philip Lane"],
        "title": ["Monetary Policy Speech", "Economic Outlook"],
        "subtitle": ["January Meeting", "February Update"],
        "contents": [
            "The ECB reaffirms its commitment to price stability." * 20,
            "Economic conditions remain challenging." * 20,
        ],
    }
    df = pd.DataFrame(csv_data)
    csv_path = tmp_path / "speeches.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_load_speeches_csv(sample_csv):
    """Test loading speeches from CSV and converting to Documents."""
    documents = load_speeches_csv(str(sample_csv))
    
    assert len(documents) == 2
    assert all(hasattr(d, "page_content") for d in documents)
    assert all(hasattr(d, "metadata") for d in documents)
    assert documents[0].metadata["speakers"] == "Christine Lagarde"
    assert documents[1].metadata["year"] == "2023"


def test_chunk_documents(sample_csv):
    """Test that chunking preserves metadata."""
    documents = load_speeches_csv(str(sample_csv))
    chunks = chunk_documents(documents, chunk_size=100, chunk_overlap=10)
    
    assert len(chunks) > len(documents)
    assert all("speakers" in c.metadata for c in chunks)


@patch("rag.ingest.OllamaEmbeddings")
@patch("rag.ingest.Chroma.from_documents")
def test_ingest_to_chromadb(mock_chroma, mock_embeddings, sample_csv):
    """Test end-to-end ingestion with mocked dependencies."""
    mock_chroma.return_value = MagicMock()
    
    result = ingest_to_chromadb(
        csv_path=str(sample_csv),
        chroma_dir="./test_chroma",
        chunk_size=500,
    )
    
    assert mock_chroma.called


def test_ingest_to_chromadb_missing_file():
    """Test that ingestion fails if CSV doesn't exist."""
    with pytest.raises(FileNotFoundError):
        ingest_to_chromadb(csv_path="./nonexistent.csv")


def test_ingest_to_chromadb_empty_csv(tmp_path):
    """Test that ingestion fails if CSV contains no valid content."""
    empty_csv = tmp_path / "empty.csv"
    pd.DataFrame({"contents": [None, None]}).to_csv(empty_csv, index=False)
    
    with pytest.raises(ValueError):
        ingest_to_chromadb(csv_path=str(empty_csv))
