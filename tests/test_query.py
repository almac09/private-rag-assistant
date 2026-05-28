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


def _make_mock_vectorstore(docs=None, score=0.85):
    if docs is None:
        docs = [_make_doc()]
    mock_vs = MagicMock()
    # similarity_search_with_relevance_scores returns [(doc, score), ...]
    mock_vs.similarity_search_with_relevance_scores.return_value = [
        (doc, score) for doc in docs
    ]
    return mock_vs


def _mock_chain_result(answer="The ECB held rates steady.", docs=None, scores=None):
    if docs is None:
        docs = [_make_doc()]
    if scores is None:
        scores = [0.85] * len(docs)
    return {
        "answer": answer,
        "source_docs": docs,
        "scores": scores,
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
def test_build_rag_chain_uses_similarity_search_with_scores(mock_ollama):
    """build_rag_chain retrieves via similarity_search_with_relevance_scores, not as_retriever."""
    from rag.query import build_rag_chain
    mock_vs = _make_mock_vectorstore()

    chain = build_rag_chain(mock_vs, model="llama3.2", k=6)

    # Chain should exist; as_retriever is not called in the new implementation
    assert chain is not None
    mock_vs.as_retriever.assert_not_called()


@patch("rag.query.ChatOllama")
def test_build_rag_chain_passes_k_to_search(mock_ollama):
    """build_rag_chain passes k to similarity_search_with_relevance_scores.

    MagicMock is not a Runnable subclass so LCEL wraps it in RunnableLambda and
    calls llm(input) — not llm.invoke(input). Set return_value.return_value so the
    callable returns a proper AIMessage that StrOutputParser can handle.
    """
    from langchain_core.messages import AIMessage
    from rag.query import build_rag_chain

    mock_vs = _make_mock_vectorstore()
    mock_ollama.return_value.return_value = AIMessage(content="Mocked answer.")

    chain = build_rag_chain(mock_vs, k=7)
    chain.invoke("test question")

    mock_vs.similarity_search_with_relevance_scores.assert_called_once_with("test question", k=7)


@patch("rag.query.ChatOllama")
def test_build_rag_chain_filters_by_score_threshold(mock_ollama):
    """Chunks below score_threshold are excluded from the retrieved set."""
    from langchain_core.messages import AIMessage
    from rag.query import build_rag_chain

    mock_vs = MagicMock()
    doc_high = _make_doc(content="High relevance.")
    doc_low = _make_doc(content="Low relevance.")
    mock_vs.similarity_search_with_relevance_scores.return_value = [
        (doc_high, 0.9), (doc_low, 0.2),
    ]
    mock_ollama.return_value.return_value = AIMessage(content="Mocked answer.")

    chain = build_rag_chain(mock_vs, score_threshold=0.5)
    result = chain.invoke("Q?")

    assert len(result["source_docs"]) == 1
    assert result["source_docs"][0].page_content == "High relevance."


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


def test_confidence_signal_scores_populate_top_and_avg():
    from rag.query import confidence_signal
    sources = [{"source": "a.csv"}, {"source": "b.csv"}]
    sig = confidence_signal("Answer.", sources, scores=[0.9, 0.7])
    assert sig["top_score"] == 0.9
    assert sig["avg_score"] == 0.8


def test_confidence_signal_no_scores_gives_none():
    from rag.query import confidence_signal
    sig = confidence_signal("Answer.", [{"source": "a.csv"}])
    assert sig["top_score"] is None
    assert sig["avg_score"] is None


def test_confidence_signal_high_level():
    from rag.query import confidence_signal
    sig = confidence_signal("Answer.", [{"source": "a.csv"}], scores=[0.85])
    assert sig["confidence_level"] == "High"


def test_confidence_signal_medium_level():
    from rag.query import confidence_signal
    sig = confidence_signal("Answer.", [{"source": "a.csv"}], scores=[0.65])
    assert sig["confidence_level"] == "Medium"


def test_confidence_signal_low_level():
    from rag.query import confidence_signal
    sig = confidence_signal("Answer.", [{"source": "a.csv"}], scores=[0.45])
    assert sig["confidence_level"] == "Low"


def test_confidence_signal_very_low_level_no_sources():
    from rag.query import confidence_signal
    sig = confidence_signal("Answer.", [])
    assert sig["confidence_level"] == "Very Low"


def test_confidence_signal_used_fallback_false_by_default():
    from rag.query import confidence_signal
    sig = confidence_signal("Answer.", [{"source": "a.csv"}])
    assert sig["used_fallback"] is False


def test_confidence_signal_used_fallback_propagated():
    from rag.query import confidence_signal
    sig = confidence_signal("Answer.", [], used_fallback=True)
    assert sig["used_fallback"] is True


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


def test_ask_returns_scores_key():
    """ask() returns a 'scores' list alongside sources."""
    from rag.query import ask
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = _mock_chain_result(scores=[0.9, 0.7])

    result = ask(mock_chain, "Q?")

    assert "scores" in result
    assert result["scores"] == [0.9, 0.7]


def test_ask_scores_populate_confidence_signal():
    """Scores passed through chain result reach confidence_signal."""
    from rag.query import ask
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = _mock_chain_result(scores=[0.85])

    result = ask(mock_chain, "Q?")

    assert result["confidence"]["top_score"] == 0.85
    assert result["confidence"]["confidence_level"] == "High"


def test_ask_fallback_fires_on_abstain():
    """When fallback_llm is provided and the answer is the abstain phrase, fallback is used."""
    from rag.query import ask, ABSTAIN_PHRASE
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = _mock_chain_result(answer=ABSTAIN_PHRASE, docs=[], scores=[])
    mock_fallback = MagicMock()
    mock_fallback.invoke.return_value = MagicMock(content="General knowledge answer.")

    result = ask(mock_chain, "Q?", fallback_llm=mock_fallback)

    assert ABSTAIN_PHRASE not in result["answer"]
    assert "general knowledge" in result["answer"].lower()
    assert result["confidence"]["used_fallback"] is True


def test_ask_fallback_not_fired_without_fallback_llm():
    """When no fallback_llm is given, abstain phrase stays in the answer."""
    from rag.query import ask, ABSTAIN_PHRASE
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = _mock_chain_result(answer=ABSTAIN_PHRASE, docs=[], scores=[])

    result = ask(mock_chain, "Q?")

    assert ABSTAIN_PHRASE in result["answer"]
    assert result["confidence"]["used_fallback"] is False


def test_ask_fallback_not_fired_for_good_answer():
    """Fallback LLM is not called when the RAG answer does not contain the abstain phrase."""
    from rag.query import ask
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = _mock_chain_result(answer="The rate is 2%.")
    mock_fallback = MagicMock()

    result = ask(mock_chain, "Q?", fallback_llm=mock_fallback)

    mock_fallback.invoke.assert_not_called()
    assert result["confidence"]["used_fallback"] is False
