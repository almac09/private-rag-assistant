"""RAG query pipeline: retrieve relevant chunks, generate answer with source citations."""

from __future__ import annotations

from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_ollama import ChatOllama

from rag.ingest import load_chromadb


_PROMPT_TEMPLATE = """\
Answer the question using ONLY the context excerpts below.
If the answer cannot be found in the context, reply: \
"I don't know based on the available speeches."

Context:
{context}

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


def build_rag_chain(vectorstore, model: str = "llama3.2", k: int = 4):
    """Build a LangChain LCEL RAG chain.

    The chain retrieves the top-*k* chunks most similar to the question, injects
    them into a prompt that constrains the LLM to the retrieved context only, and
    returns both the generated answer and the source documents — without hitting
    the vectorstore twice.

    Args:
        vectorstore: ChromaDB vectorstore (from :func:`load_vectorstore`).
        model: Ollama model name to use for generation.
        k: Number of chunks to retrieve per query.

    Returns:
        LangChain runnable — invoke with a question string; returns a dict with
        keys ``answer`` (str), ``source_docs`` (list of Documents), ``context``
        (str), and ``question`` (str).
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    prompt = ChatPromptTemplate.from_template(_PROMPT_TEMPLATE)
    llm = ChatOllama(model=model)

    chain = (
        RunnableParallel(source_docs=retriever, question=RunnablePassthrough())
        .assign(context=lambda x: _format_docs(x["source_docs"]))
        .assign(answer=prompt | llm | StrOutputParser())
    )
    return chain


def ask(chain, question: str, *, log_path: str | Path | None = None) -> dict:
    """Run a question through the RAG chain and return answer + sources.

    Args:
        chain: LangChain RAG chain from :func:`build_rag_chain`.
        question: Natural-language question.
        log_path: If provided, append the query result to this JSONL file via
            :func:`rag.logger.log_query`.  Failed queries are flagged
            automatically so they can be replayed later.

    Returns:
        Dict with keys ``answer`` (str) and ``sources`` (list of metadata dicts,
        one per retrieved chunk).
    """
    result = chain.invoke(question)
    out = {
        "answer": result["answer"],
        "sources": [doc.metadata for doc in result["source_docs"]],
    }
    if log_path is not None:
        from rag.logger import log_query

        log_query(question, out["answer"], out["sources"], path=log_path)
    return out
