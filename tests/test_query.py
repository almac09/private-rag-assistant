"""
Tests for the RAG query pipeline.

Unit tests mock ChromaDB and Ollama — run in CI with no external dependencies.
Integration tests are marked @ollama_required and skipped when Ollama is offline.
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(speaker="Christine Lagarde", title="Test Speech", content="ECB monetary policy."):
    return Document(
        page_content=content,
        metadata={
            "speakers": speaker,
            "title": title,
            "date": "2023-01-01",
            "year": "2023",
            "source": "test.csv",
        },
    )


def _make_mock_vectorstore(docs=None):
    if docs is None:
        docs = [_make_doc()]
    mock_vs = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.invoke = MagicMock(return_value=docs)
    mock_vs.as_retriever.return_value = mock_retriever
    return mock_vs


def _mock_chain_result(answer="The ECB held rates steady.", docs=None):
    if docs is None:
        docs = [_make_doc()]
    return {
        "answer": answer,
        "source_docs": docs,
        "question": "What did the ECB do?",
        "context": "\n\n".join(d.page_content for d in docs),
    }


# ---------------------------------------------------------------------------
# load_vectorstore
# ---------------------------------------------------------------------------

@patch("rag.query.load_chromadb")
def test_load_vectorstore_delegates(mock_load):
    """load_vectorstore delegates to load_chromadb with correct arguments."""
    from rag.query import load_vectorstore
    mock_load.return_value = MagicMock()

    load_vectorstore(persist_dir="my_chroma", ollama_model="nomic-embed-text")

    mock_load.assert_called_once_with(chroma_dir="my_chroma", ollama_model="nomic-embed-text")


@patch("rag.query.load_chromadb")
def test_load_vectorstore_default_args(mock_load):
    """load_vectorstore uses 'chroma_db' as default persist directory."""
    from rag.query import load_vectorstore
    mock_load.return_value = MagicMock()

    load_vectorstore()

    assert mock_load.call_args.kwargs["chroma_dir"] == "chroma_db"


@patch("rag.query.load_chromadb")
def test_load_vectorstore_returns_vectorstore(mock_load):
    """load_vectorstore returns whatever load_chromadb returns."""
    from rag.query import load_vectorstore
    sentinel = MagicMock(name="vectorstore")
    mock_load.return_value = sentinel

    result = load_vectorstore()

    assert result is sentinel


# ---------------------------------------------------------------------------
# build_rag_chain
# ---------------------------------------------------------------------------

@patch("rag.query.ChatOllama")
def test_build_rag_chain_returns_chain(mock_ollama):
    """build_rag_chain returns a runnable object."""
    from rag.query import build_rag_chain
    mock_vs = _make_mock_vectorstore()

    chain = build_rag_chain(mock_vs, model="llama3.2", k=4)

    assert chain is not None


@patch("rag.query.ChatOllama")
def test_build_rag_chain_configures_retriever(mock_ollama):
    """build_rag_chain calls as_retriever with the correct k value."""
    from rag.query import build_rag_chain
    mock_vs = _make_mock_vectorstore()

    build_rag_chain(mock_vs, model="llama3.2", k=6)

    mock_vs.as_retriever.assert_called_once_with(search_kwargs={"k": 6})


@patch("rag.query.ChatOllama")
def test_build_rag_chain_uses_model(mock_ollama):
    """build_rag_chain passes model name through to ChatOllama."""
    from rag.query import build_rag_chain
    mock_vs = _make_mock_vectorstore()

    build_rag_chain(mock_vs, model="mistral")

    mock_ollama.assert_called_once_with(model="mistral")


# ---------------------------------------------------------------------------
# confidence_signal
# ---------------------------------------------------------------------------

def test_confidence_signal_certain_answer():
    from rag.query import confidence_signal
    sources = [{"source": "s.txt"}, {"source": "t.txt"}]
    sig = confidence_signal("The ECB raised rates by 25bp.", sources)
    assert sig["is_uncertain"] is False
    assert sig["n_sources"] == 2
    assert sig["has_sources"] is True
    assert sig["unique_documents"] == 2


def test_confidence_signal_uncertain_answer():
    from rag.query import confidence_signal, ABSTAIN_PHRASE
    sig = confidence_signal(ABSTAIN_PHRASE, [])
    assert sig["is_uncertain"] is True
    assert sig["n_sources"] == 0
    assert sig["has_sources"] is False


def test_confidence_signal_counts_unique_sources():
    from rag.query import confidence_signal
    # Two chunks from same file → unique_documents == 1
    sources = [{"source": "speech.txt"}, {"source": "speech.txt"}]
    sig = confidence_signal("Some answer.", sources)
    assert sig["unique_documents"] == 1


def test_confidence_signal_handles_missing_source_key():
    from rag.query import confidence_signal
    # Metadata without 'source' key should not crash
    sources = [{"speaker": "Lagarde"}, {"speaker": "Lane"}]
    sig = confidence_signal("Answer.", sources)
    assert sig["unique_documents"] == 0
    assert sig["n_sources"] == 2


def test_confidence_signal_empty_sources():
    from rag.query import confidence_signal
    sig = confidence_signal("Answer with no sources.", [])
    assert sig["n_sources"] == 0
    assert sig["has_sources"] is False
    assert sig["unique_documents"] == 0


# ---------------------------------------------------------------------------
# ask — return shape
# ---------------------------------------------------------------------------

def test_ask_returns_answer_sources_confidence():
    """ask() returns a dict with 'answer', 'sources', and 'confidence' keys."""
    from rag.query import ask
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = _mock_chain_result()

    result = ask(mock_chain, "What did the ECB do?")

    assert "answer" in result
    assert "sources" in result
    assert "confidence" in result


def test_ask_answer_value():
    """ask() surfaces the answer string from the chain result."""
    from rag.query import ask
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = _mock_chain_result(answer="Rates were raised.")

    result = ask(mock_chain, "Q?")

    assert result["answer"] == "Rates were raised."


def test_ask_sources_length():
    """ask() returns one source entry per retrieved document."""
    from rag.query import ask
    docs = [_make_doc("Lagarde"), _make_doc("Lane"), _make_doc("Schnabel")]
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = _mock_chain_result(docs=docs)

    result = ask(mock_chain, "Q?")

    assert len(result["sources"]) == 3


def test_ask_sources_are_metadata_dicts():
    """ask() converts source Documents to metadata dicts."""
    from rag.query import ask
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = _mock_chain_result()

    result = ask(mock_chain, "Q?")

    assert all(isinstance(s, dict) for s in result["sources"])


def test_ask_sources_contain_required_keys():
    """Each source dict has the expected metadata keys."""
    from rag.query import ask
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = _mock_chain_result()

    result = ask(mock_chain, "Q?")

    required = {"speakers", "title", "date", "year", "source"}
    for src in result["sources"]:
        missing = required - src.keys()
        assert not missing, f"Source missing keys: {missing}"


def test_ask_passes_question_to_chain():
    """ask() passes the question string verbatim to chain.invoke."""
    from rag.query import ask
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = _mock_chain_result()

    ask(mock_chain, "What is the inflation target?")

    mock_chain.invoke.assert_called_once_with("What is the inflation target?")


def test_ask_sources_speaker_metadata():
    """ask() preserves speaker name in returned source metadata."""
    from rag.query import ask
    docs = [_make_doc(speaker="Philip Lane", title="Economic Outlook")]
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = _mock_chain_result(docs=docs)

    result = ask(mock_chain, "Q?")

    assert result["sources"][0]["speakers"] == "Philip Lane"


def test_ask_confidence_is_certain_for_good_answer():
    """confidence.is_uncertain is False when the model gives a real answer."""
    from rag.query import ask
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = _mock_chain_result(answer="The rate is 2%.")

    result = ask(mock_chain, "Q?")

    assert result["confidence"]["is_uncertain"] is False


def test_ask_confidence_is_uncertain_for_abstain():
    """confidence.is_uncertain is True when model returns the abstain phrase."""
    from rag.query import ask, ABSTAIN_PHRASE
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = _mock_chain_result(answer=ABSTAIN_PHRASE)

    result = ask(mock_chain, "Q?")

    assert result["confidence"]["is_uncertain"] is True


def test_ask_confidence_n_sources_matches_docs():
    """confidence.n_sources equals the number of retrieved documents."""
    from rag.query import ask
    docs = [_make_doc(), _make_doc(), _make_doc()]
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = _mock_chain_result(docs=docs)

    result = ask(mock_chain, "Q?")

    assert result["confidence"]["n_sources"] == 3
