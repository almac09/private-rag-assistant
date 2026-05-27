"""
Tests for document ingestion pipeline (ECB speeches CSV).

Unit tests use a synthetic CSV fixture and run in CI.
Integration tests (marked real_data) require the actual ECB speeches CSV at
inputs/Data/all_ECB_speeches_csv.csv and are skipped in CI automatically.
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from rag.ingest import load_speeches_csv, chunk_documents, ingest_to_chromadb

REAL_CSV = Path("inputs/Data/all_ECB_speeches_csv.csv")
real_data = pytest.mark.skipif(not REAL_CSV.exists(), reason="real ECB speeches CSV not available in CI")


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


# ---------------------------------------------------------------------------
# Integration / smoke tests — skipped in CI, run locally against real data
# ---------------------------------------------------------------------------

@real_data
def test_real_csv_loads():
    """Real CSV loads without error and returns a non-trivial number of speeches."""
    docs = load_speeches_csv(str(REAL_CSV))
    assert len(docs) > 100, f"Expected >100 speeches, got {len(docs)}"


@real_data
def test_real_csv_metadata_shape():
    """Every document from the real CSV has the expected metadata keys."""
    docs = load_speeches_csv(str(REAL_CSV))
    required_keys = {"date", "year", "speakers", "title", "source"}
    for doc in docs[:10]:
        missing = required_keys - doc.metadata.keys()
        assert not missing, f"Missing metadata keys: {missing}"


@real_data
def test_real_csv_chunking():
    """Chunking the first 5 real speeches produces more chunks than documents."""
    docs = load_speeches_csv(str(REAL_CSV))[:5]
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
    assert len(chunks) > len(docs)
    assert all("speakers" in c.metadata for c in chunks)


# ---------------------------------------------------------------------------
# Issue #4 — Chunk quality spot-checks
# ---------------------------------------------------------------------------

def test_chunk_quality_no_empty_chunks(sample_csv):
    """No chunk should be empty or whitespace-only."""
    docs = load_speeches_csv(str(sample_csv))
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
    empty = [c for c in chunks if not c.page_content.strip()]
    assert not empty, f"{len(empty)} empty chunk(s) found"


def test_chunk_quality_minimum_length(sample_csv):
    """All chunks should be at least 10 characters (no degenerate splits)."""
    docs = load_speeches_csv(str(sample_csv))
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
    short = [c for c in chunks if len(c.page_content.strip()) < 10]
    assert not short, f"{len(short)} chunk(s) shorter than 10 chars"


def test_chunk_quality_metadata_preserved(sample_csv):
    """Every chunk must carry the full set of required metadata keys."""
    required = {"date", "year", "speakers", "title", "source"}
    docs = load_speeches_csv(str(sample_csv))
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
    for chunk in chunks:
        missing = required - chunk.metadata.keys()
        assert not missing, f"Chunk missing metadata keys: {missing}"


def test_chunk_quality_respects_chunk_size(sample_csv):
    """No chunk should significantly exceed the requested chunk_size."""
    chunk_size = 200
    docs = load_speeches_csv(str(sample_csv))
    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=20)
    oversized = [c for c in chunks if len(c.page_content) > chunk_size * 1.1]
    assert not oversized, f"{len(oversized)} chunk(s) exceed chunk_size by >10%"


@real_data
def test_real_chunk_quality():
    """Spot-check chunk quality against the real ECB speeches corpus."""
    docs = load_speeches_csv(str(REAL_CSV))[:20]
    chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=200)

    empty = [c for c in chunks if not c.page_content.strip()]
    short = [c for c in chunks if len(c.page_content.strip()) < 10]
    required = {"date", "year", "speakers", "title", "source"}
    bad_meta = [c for c in chunks if required - c.metadata.keys()]

    assert not empty, f"{len(empty)} empty chunks in real corpus"
    assert not short, f"{len(short)} degenerate chunks in real corpus"
    assert not bad_meta, f"{len(bad_meta)} chunks with missing metadata"
