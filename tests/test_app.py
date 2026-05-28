"""Tests for the Streamlit demo app (app.py).

Two test groups:
  1. Source-inspection tests (CI-safe, no Ollama/ChromaDB).
  2. AppTest render tests (CI-safe — all heavy resources are mocked).
  3. rag.query integration tests used by the app (CI-safe, mocked chain).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).parent.parent
APP_PATH = str(ROOT / "app.py")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_doc(
    content="Inflation in the euro area fell to 2.9% in December 2023.",
    source="cbi_speeches.csv",
    speaker="Governor Lane",
    date="2023-06-01",
    title="Annual Report 2023",
):
    doc = MagicMock()
    doc.page_content = content
    doc.metadata = {"source": source, "speaker": speaker, "date": date, "title": title}
    return doc


def _make_mock_chain(answer="Inflation fell to 2.9% in December 2023.", n_docs=2):
    docs = [_make_doc() for _ in range(n_docs)]
    chain = MagicMock()
    chain.invoke.return_value = {
        "answer": answer,
        "source_docs": docs,
        "context": "Some speech context.",
        "question": "",
    }
    return chain


def _make_mock_llm(content="The ECB raised rates several times in 2023."):
    response = MagicMock()
    response.content = content
    llm = MagicMock()
    llm.invoke.return_value = response
    return llm


def _app_test_with_mocks(chain=None, llm=None) -> AppTest:
    """Return an AppTest instance with Ollama and ChromaDB mocked out."""
    if chain is None:
        chain = _make_mock_chain()
    if llm is None:
        llm = _make_mock_llm()

    mock_vs = MagicMock()

    with (
        patch("rag.query.load_vectorstore", return_value=mock_vs),
        patch("rag.query.build_rag_chain", return_value=chain),
        patch("langchain_ollama.ChatOllama", return_value=llm),
        # Mock the model list so the sidebar selectbox renders without calling Ollama
        patch("app._list_chat_models", return_value=["llama3.2:1b", "llama3:latest"]),
    ):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
    return at


# ---------------------------------------------------------------------------
# Group 1: Source-inspection tests
# ---------------------------------------------------------------------------


def test_app_file_exists():
    assert (ROOT / "app.py").exists()


def test_app_imports_from_rag_query():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "from rag.query import" in src


def test_app_calls_ask():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "ask(" in src


def test_app_calls_build_rag_chain():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "build_rag_chain" in src


def test_app_calls_load_vectorstore():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "load_vectorstore" in src


def test_app_uses_cache_resource():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "cache_resource" in src


def test_app_queries_ollama_for_model_list():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "_list_chat_models" in src or "ollama.list" in src or "_ollama.list" in src


def test_app_handles_uncertain_answers():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "is_uncertain" in src


# ---------------------------------------------------------------------------
# Group 2: AppTest render tests (CI-safe — mocked resources)
# ---------------------------------------------------------------------------


def test_app_renders_without_exception():
    """App loads its initial state without raising any exception."""
    at = _app_test_with_mocks()
    assert not at.exception, f"App raised: {at.exception}"


def test_app_title_is_present():
    at = _app_test_with_mocks()
    titles = [t.value for t in at.title]
    assert any("RAG" in t for t in titles)


def test_app_has_question_text_input():
    at = _app_test_with_mocks()
    assert len(at.text_input) >= 1, "App must render a text input for the question"


def test_app_has_ask_button():
    at = _app_test_with_mocks()
    assert len(at.button) >= 1, "App must render an Ask button"


def test_app_ask_button_disabled_when_no_question():
    """Ask button should be disabled when the question field is empty."""
    at = _app_test_with_mocks()
    ask_button = at.button[0]
    assert ask_button.disabled, "Ask button must be disabled with no question entered"


def test_app_shows_answers_after_question_submitted():
    """After entering a question and clicking Ask, both panels produce markdown output."""
    chain = _make_mock_chain(answer="Inflation fell to 2.9%.")
    llm = _make_mock_llm(content="Inflation declined significantly in 2023.")
    mock_vs = MagicMock()

    with (
        patch("rag.query.load_vectorstore", return_value=mock_vs),
        patch("rag.query.build_rag_chain", return_value=chain),
        patch("langchain_ollama.ChatOllama", return_value=llm),
        patch("app._list_chat_models", return_value=["llama3.2:1b"]),
    ):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.text_input[0].set_value("What happened to inflation in 2023?").run()
        at.button[0].click().run()

    assert not at.exception, f"App raised after submit: {at.exception}"
    all_text = " ".join(m.value for m in at.markdown)
    assert "2.9%" in all_text or "declined" in all_text, (
        "At least one answer should appear in the rendered markdown"
    )


def test_app_shows_error_gracefully_when_resources_fail():
    """If Ollama/ChromaDB fails to load, the app shows an error without crashing.

    st.cache_resource persists across AppTest instances in the same pytest session,
    so we clear it before this test to force the patched side_effect to actually run.
    """
    import streamlit as st

    st.cache_resource.clear()

    with (
        patch("rag.query.load_vectorstore", side_effect=ConnectionError("Ollama not running")),
        patch("langchain_ollama.ChatOllama", side_effect=ConnectionError("Ollama not running")),
        patch("app._list_chat_models", return_value=["llama3.2:1b"]),
    ):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()

    # st.stop() is called after st.error() — AppTest handles this gracefully
    assert not at.exception, f"App crashed instead of showing error: {at.exception}"
    error_msgs = [e.value for e in at.error]
    assert len(error_msgs) > 0, "App must render an error message when resources fail"


# ---------------------------------------------------------------------------
# Group 3: rag.query integration used by the app (CI-safe, mocked chain)
# ---------------------------------------------------------------------------


def _make_mock_chain_raw(answer="The inflation rate was 3%.", docs=None):
    if docs is None:
        docs = [_make_doc()]
    chain = MagicMock()
    chain.invoke.return_value = {"answer": answer, "source_docs": docs, "context": "", "question": ""}
    return chain


def test_ask_returns_expected_keys():
    from rag.query import ask

    result = ask(_make_mock_chain_raw(), "What is inflation?")
    assert set(result.keys()) >= {"answer", "sources", "confidence"}


def test_ask_confidence_certain_for_real_answer():
    from rag.query import ask

    result = ask(_make_mock_chain_raw(answer="The inflation rate was 3%."), "Inflation?")
    assert result["confidence"]["is_uncertain"] is False


def test_ask_confidence_uncertain_for_abstain():
    from rag.query import ABSTAIN_PHRASE, ask

    result = ask(_make_mock_chain_raw(answer=ABSTAIN_PHRASE), "Unknown topic")
    assert result["confidence"]["is_uncertain"] is True


def test_ask_n_sources_matches_retrieved_docs():
    from rag.query import ask

    docs = [_make_doc(source=f"doc_{i}.csv") for i in range(3)]
    result = ask(_make_mock_chain_raw(docs=docs), "Question")
    assert result["confidence"]["n_sources"] == 3


def test_ask_unique_documents_counted_correctly():
    from rag.query import ask

    docs = [_make_doc(source="a.csv"), _make_doc(source="a.csv"), _make_doc(source="b.csv")]
    result = ask(_make_mock_chain_raw(docs=docs), "Question")
    assert result["confidence"]["unique_documents"] == 2


def test_ask_sources_are_metadata_dicts():
    from rag.query import ask

    result = ask(_make_mock_chain_raw(), "Question")
    assert isinstance(result["sources"], list)
    assert all(isinstance(s, dict) for s in result["sources"])
