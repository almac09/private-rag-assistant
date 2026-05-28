"""RAG query pipeline: retrieve relevant chunks, generate answer with source citations."""

from __future__ import annotations

from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_ollama import ChatOllama

from rag.ingest import load_chromadb


# Phrase the LLM is instructed to return when context contains no useful answer.
# Used by confidence_signal() and rag.logger.is_failed_query().
ABSTAIN_PHRASE = "I don't know based on the available speeches."

_PROMPT_TEMPLATE = f"""\
Answer the question using ONLY the context excerpts below.
If the answer cannot be found in the context, reply: "{ABSTAIN_PHRASE}"

Context:
{{context}}

Question: {{question}}

Answer:"""

_FALLBACK_PROMPT = """\
Answer the following question using your general knowledge.
Be concise and factual.

Question: {question}

Answer:"""


def _format_docs(docs: list) -> str:
    return "\n\n".join(d.page_content for d in docs)


def load_vectorstore(
    persist_dir: str | Path = "chroma_db",
    ollama_model: str = "nomic-embed-text",
):
    """Load an existing ChromaDB vectorstore from disk.

    Args:
        persist_dir: Directory where ChromaDB was persisted by
            :func:`rag.ingest.ingest_to_chromadb`.
        ollama_model: Embedding model name (must match the model used at ingest time).

    Returns:
        ChromaDB vectorstore instance ready for similarity search.
    """
    return load_chromadb(chroma_dir=str(persist_dir), ollama_model=ollama_model)


def build_rag_chain(vectorstore, model: str = "llama3.2", k: int = 4, score_threshold: float = 0.0):
    """Build a LangChain LCEL RAG chain with optional similarity-score filtering.

    Retrieves the top-*k* chunks most similar to the question via
    ``similarity_search_with_relevance_scores``, optionally filtering out chunks
    below *score_threshold*, then injects them into a grounded prompt and generates
    an answer. Both the generated answer and the retrieved (doc, score) pairs are
    returned in a single pass.

    Args:
        vectorstore: ChromaDB vectorstore (from :func:`load_vectorstore`).
        model: Ollama model name to use for generation.
        k: Number of chunks to retrieve per query.
        score_threshold: Minimum relevance score (0-1) for a chunk to be included.
            0.0 means no filtering.

    Returns:
        LangChain runnable. Invoke with a question string; returns a dict with keys
        ``answer`` (str), ``source_docs`` (list of Documents), ``scores`` (list of
        float), ``context`` (str), and ``question`` (str).
    """

    def _retrieve(question: str):
        results = vectorstore.similarity_search_with_relevance_scores(question, k=k)
        if score_threshold > 0.0:
            results = [(doc, score) for doc, score in results if score >= score_threshold]
        return results

    prompt = ChatPromptTemplate.from_template(_PROMPT_TEMPLATE)
    llm = ChatOllama(model=model)

    chain = (
        RunnableParallel(
            scored_docs=RunnableLambda(_retrieve),
            question=RunnablePassthrough(),
        )
        .assign(source_docs=lambda x: [doc for doc, _ in x["scored_docs"]])
        .assign(scores=lambda x: [score for _, score in x["scored_docs"]])
        .assign(context=lambda x: _format_docs(x["source_docs"]))
        .assign(answer=prompt | llm | StrOutputParser())
    )
    return chain


def confidence_signal(
    answer: str,
    sources: list[dict],
    scores: list[float] | None = None,
    used_fallback: bool = False,
) -> dict:
    """Return a lightweight confidence estimate based on answer, sources, and scores.

    No additional LLM calls are made; this is a heuristic only.

    Signals:

    - ``is_uncertain``: ``True`` when the model admitted it could not find the answer
      (detected via :data:`ABSTAIN_PHRASE`).
    - ``n_sources``: number of chunks retrieved.
    - ``has_sources``: ``True`` when at least one chunk was retrieved.
    - ``unique_documents``: number of distinct source documents.
    - ``top_score``: highest similarity score across retrieved chunks, or ``None``.
    - ``avg_score``: mean similarity score, or ``None``.
    - ``confidence_level``: categorical label — ``"High"`` (top_score >= 0.8),
      ``"Medium"`` (>= 0.6), ``"Low"`` (>= 0.4), ``"Very Low"`` otherwise.
    - ``used_fallback``: ``True`` when the always-answer fallback fired.

    Args:
        answer: The answer string returned by :func:`ask`.
        sources: The list of metadata dicts returned by :func:`ask`.
        scores: Optional list of relevance scores parallel to *sources*.
        used_fallback: Pass ``True`` when the always-answer fallback was used.

    Returns:
        Dict with confidence fields described above.
    """
    is_uncertain = ABSTAIN_PHRASE in answer
    n_sources = len(sources)
    unique_docs = len({s.get("source", "") for s in sources if s.get("source")})

    top_score = round(max(scores), 3) if scores else None
    avg_score = round(sum(scores) / len(scores), 3) if scores else None

    if is_uncertain or n_sources == 0:
        level = "Very Low"
    elif top_score is None:
        level = "Unknown"
    elif top_score >= 0.8:
        level = "High"
    elif top_score >= 0.6:
        level = "Medium"
    elif top_score >= 0.4:
        level = "Low"
    else:
        level = "Very Low"

    return {
        "is_uncertain": is_uncertain,
        "n_sources": n_sources,
        "has_sources": n_sources > 0,
        "unique_documents": unique_docs,
        "top_score": top_score,
        "avg_score": avg_score,
        "confidence_level": level,
        "used_fallback": used_fallback,
    }


def ask(
    chain,
    question: str,
    *,
    log_path: str | Path | None = None,
    fallback_llm=None,
) -> dict:
    """Run a question through the RAG chain and return answer, sources, scores, and confidence.

    Args:
        chain: LangChain RAG chain from :func:`build_rag_chain`.
        question: Natural-language question.
        log_path: If provided, append the query result to this JSONL file via
            :func:`rag.logger.log_query`.
        fallback_llm: Optional ``ChatOllama`` instance. When provided and the RAG chain
            returns :data:`ABSTAIN_PHRASE`, a second call is made without the retrieval
            constraint. The fallback answer is clearly labelled so users know it is not
            grounded in the corpus.

    Returns:
        Dict with keys ``answer`` (str), ``sources`` (list of metadata dicts),
        ``scores`` (list of float), and ``confidence`` (dict from
        :func:`confidence_signal`).
    """
    result = chain.invoke(question)
    sources = [doc.metadata for doc in result["source_docs"]]
    scores = result.get("scores", [])
    answer = result["answer"]
    used_fallback = False

    if ABSTAIN_PHRASE in answer and fallback_llm is not None:
        fallback_response = fallback_llm.invoke(question)
        fallback_text = (
            fallback_response.content
            if hasattr(fallback_response, "content")
            else str(fallback_response)
        )
        answer = (
            f"*Note: No relevant answer found in the speeches corpus. "
            f"The following uses general knowledge only:*\n\n{fallback_text}"
        )
        used_fallback = True

    out = {
        "answer": answer,
        "sources": sources,
        "scores": scores,
        "confidence": confidence_signal(answer, sources, scores, used_fallback),
    }
    if log_path is not None:
        from rag.logger import log_query

        log_query(question, answer, sources, path=log_path)
    return out
