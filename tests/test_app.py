"""Tests for the Streamlit demo app (app.py).

All tests are CI-safe: no Ollama, no real ChromaDB required.
We verify module structure and that the underlying rag.* functions
used by the app behave correctly with mocked chains.
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent


def _make_mock_doc(content="Test content.", source="speech_001.csv", speaker="Governor"):
    doc = MagicMock()
    doc.page_content = content
    doc.metadata = {"source": source, "speaker": speaker, "date": "2023-01-01"}
    return doc


def _make_mock_chain(answer="The inflation rate was 3%.", docs=None):
    if docs is None:
        docs = [_make_mock_doc()]
    chain = MagicMock()
    chain.invoke.return_value = {"answer": answer, "source_docs": docs, "context": "", "question": ""}
    return chain


# ---------------------------------------------------------------------------
# app.py existence and importability
# ---------------------------------------------------------------------------


def test_app_file_exists():
    assert (ROOT / "app.py").exists(), "app.py must exist in the repo root"


def test_app_imports_rag_modules():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "from rag.query import" in source, "app.py must import from rag.query"


def test_app_uses_ask():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "ask(" in source, "app.py must call ask()"


def test_app_uses_build_rag_chain():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "build_rag_chain" in source


def test_app_uses_load_vectorstore():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "load_vectorstore" in source


def test_app_uses_cache_resource():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "cache_resource" in source, "app.py should cache expensive resources"


def test_app_handles_abstain_phrase():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "ABSTAIN_PHRASE" in source or "is_uncertain" in source, (
        "app.py must handle the uncertain/abstain case"
    )


# ---------------------------------------------------------------------------
# Underlying rag.query functions used by the app
# ---------------------------------------------------------------------------


def test_ask_returns_expected_keys():
    from rag.query import ask

    chain = _make_mock_chain()
    result = ask(chain, "What is inflation?")
    assert set(result.keys()) >= {"answer", "sources", "confidence"}


def test_ask_confidence_is_certain_for_real_answer():
    from rag.query import ask

    chain = _make_mock_chain(answer="The inflation rate was 3%.")
    result = ask(chain, "What is inflation?")
    assert result["confidence"]["is_uncertain"] is False


def test_ask_confidence_is_uncertain_for_abstain():
    from rag.query import ABSTAIN_PHRASE, ask

    chain = _make_mock_chain(answer=ABSTAIN_PHRASE)
    result = ask(chain, "Unknown topic")
    assert result["confidence"]["is_uncertain"] is True


def test_ask_n_sources_matches_retrieved_docs():
    from rag.query import ask

    docs = [_make_mock_doc(source=f"doc_{i}.csv") for i in range(3)]
    chain = _make_mock_chain(docs=docs)
    result = ask(chain, "Question")
    assert result["confidence"]["n_sources"] == 3


def test_ask_unique_documents_counted_correctly():
    from rag.query import ask

    docs = [
        _make_mock_doc(source="a.csv"),
        _make_mock_doc(source="a.csv"),
        _make_mock_doc(source="b.csv"),
    ]
    chain = _make_mock_chain(docs=docs)
    result = ask(chain, "Question")
    assert result["confidence"]["unique_documents"] == 2


def test_ask_sources_are_metadata_dicts():
    from rag.query import ask

    chain = _make_mock_chain()
    result = ask(chain, "Question")
    assert isinstance(result["sources"], list)
    assert all(isinstance(s, dict) for s in result["sources"])
