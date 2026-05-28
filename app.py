"""Streamlit demo: side-by-side comparison of baseline LLM vs RAG pipeline.

Run with:
    streamlit run app.py

Requires:
    - ollama serve (llama3.2 and nomic-embed-text pulled)
    - ChromaDB populated: python scripts/ollama_ready.py
"""

import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from rag.query import ABSTAIN_PHRASE, ask, build_rag_chain, load_vectorstore

st.set_page_config(
    page_title="RAG Assistant — Live Demo",
    page_icon="📚",
    layout="wide",
)

st.title("RAG Assistant — Live Demo")
st.caption("CBI Speeches corpus · llama3.2 · nomic-embed-text · ChromaDB · LangChain")

st.markdown(
    """
Ask a question about Central Bank of Ireland speeches.
**Left panel** answers using the raw LLM with no retrieval.
**Right panel** answers using the full RAG pipeline, grounded in the speeches corpus.
"""
)

st.divider()


@st.cache_resource(show_spinner="Loading RAG pipeline...")
def _get_rag_chain():
    vs = load_vectorstore()
    return build_rag_chain(vs)


@st.cache_resource(show_spinner="Connecting to Ollama...")
def _get_baseline_llm():
    return ChatOllama(model="llama3.2")


def _load_resources():
    try:
        chain = _get_rag_chain()
        llm = _get_baseline_llm()
        return chain, llm, None
    except Exception as exc:
        return None, None, str(exc)


chain, llm, load_error = _load_resources()

if load_error:
    st.error(
        f"Could not initialise pipeline: {load_error}\n\n"
        "Make sure `ollama serve` is running and the ChromaDB collection is populated "
        "(`python scripts/ollama_ready.py`)."
    )
    st.stop()

question = st.text_input(
    "Ask a question:",
    placeholder="e.g. What is the Central Bank's view on inflation in 2023?",
)

ask_btn = st.button("Ask", type="primary", disabled=not question)

if ask_btn and question:
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.subheader("Without RAG")
        st.caption("Raw LLM — no retrieval, no grounding in the corpus")
        with st.spinner("Generating baseline answer..."):
            baseline_response = llm.invoke([HumanMessage(content=question)])
            baseline_answer = baseline_response.content
        st.write(baseline_answer)

    with col_right:
        st.subheader("With RAG")
        st.caption("Retrieval-augmented — grounded in CBI speeches")
        with st.spinner("Retrieving chunks and generating answer..."):
            result = ask(chain, question)

        rag_answer = result["answer"]
        conf = result["confidence"]
        sources = result["sources"]

        st.write(rag_answer)

        if conf["is_uncertain"]:
            st.warning(
                f"Model could not find an answer in the corpus "
                f"(retrieved {conf['n_sources']} chunks)."
            )
        else:
            st.success(
                f"Grounded in **{conf['n_sources']} chunks** "
                f"from **{conf['unique_documents']} document(s)**."
            )

        if sources:
            with st.expander(f"Retrieved sources ({len(sources)} chunks)"):
                for i, src in enumerate(sources, 1):
                    title = src.get("title") or src.get("source", "unknown")
                    date = src.get("date", "")
                    speaker = src.get("speaker", "")
                    meta_parts = [p for p in [speaker, date] if p]
                    meta = " · ".join(meta_parts)
                    st.markdown(f"**{i}.** {title}")
                    if meta:
                        st.caption(meta)

st.divider()
st.caption(
    "Project: *Building a RAG System I Can Explain* · "
    "Alan McDonagh (A00052777) · TU Dublin CPD Foundations of AI"
)
