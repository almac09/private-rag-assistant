"""Streamlit demo: side-by-side comparison of baseline LLM vs RAG pipeline.

Run with:
    streamlit run app.py

Requires:
    - ollama serve (a chat model and nomic-embed-text pulled)
    - ChromaDB populated: python scripts/ollama_ready.py
"""

import time

import ollama as _ollama
import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from rag.query import ABSTAIN_PHRASE, ask, build_rag_chain, load_vectorstore

st.set_page_config(
    page_title="RAG Assistant — Live Demo",
    page_icon="📚",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Mode presets
# ---------------------------------------------------------------------------

_MODES = {
    "Standard": {
        "k": 4, "threshold": 0.0, "fallback": False,
        "show_scores": False, "show_raw_chunks": False,
    },
    "Customer-Facing": {
        "k": 3, "threshold": 0.3, "fallback": True,
        "show_scores": False, "show_raw_chunks": False,
    },
    "Pedantic": {
        "k": 6, "threshold": 0.5, "fallback": False,
        "show_scores": True, "show_raw_chunks": False,
    },
    "Debug": {
        "k": 8, "threshold": 0.0, "fallback": False,
        "show_scores": True, "show_raw_chunks": True,
    },
    "Custom": {
        "k": 4, "threshold": 0.0, "fallback": False,
        "show_scores": False, "show_raw_chunks": False,
    },
}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

_EMBED_KEYWORDS = ("embed", "nomic", "mxbai", "bge", "e5")


@st.cache_data(ttl=60)
def _list_chat_models() -> list[str]:
    """Return names of locally installed Ollama chat models."""
    try:
        response = _ollama.list()
        all_models = [m.model for m in response.models]
        return [m for m in all_models if not any(kw in m.lower() for kw in _EMBED_KEYWORDS)]
    except Exception:
        return []


with st.sidebar:
    st.header("Settings")

    # Model selection
    available_models = _list_chat_models()
    if available_models:
        selected_model = st.selectbox(
            "Ollama model",
            available_models,
            help="Both panels use this model.",
        )
    else:
        selected_model = st.text_input(
            "Ollama model",
            value="llama3.2",
            help="Could not query Ollama — enter the model name manually.",
        )

    st.divider()

    # Mode selector
    mode = st.selectbox("Mode", list(_MODES.keys()), index=0, help=(
        "Presets for k, threshold, fallback, and metadata display. "
        "Choose Custom to set each parameter individually."
    ))
    mode_cfg = _MODES[mode]

    if mode == "Custom":
        k = st.slider("Top-k chunks", 1, 10, 4, help="Number of chunks retrieved per query.")
        threshold = st.slider(
            "Similarity threshold", 0.0, 1.0, 0.0, step=0.05,
            help="Chunks below this score are excluded. 0 = no filtering.",
        )
        use_fallback = st.toggle(
            "Always answer (fallback mode)",
            value=False,
            help="If no grounded answer is found, answer from general knowledge with a disclaimer.",
        )
        show_scores = st.toggle("Show similarity scores", value=False)
        show_raw_chunks = st.toggle("Show raw chunk text", value=False)
    else:
        k = mode_cfg["k"]
        threshold = mode_cfg["threshold"]
        use_fallback = mode_cfg["fallback"]
        show_scores = mode_cfg["show_scores"]
        show_raw_chunks = mode_cfg["show_raw_chunks"]
        st.caption(
            f"k={k} · threshold={threshold:.2f} · "
            f"fallback={'on' if use_fallback else 'off'}"
        )

    st.divider()
    st.caption(
        "**Corpus:** CBI Speeches  \n"
        "**Embeddings:** nomic-embed-text  \n"
        "**Vector store:** ChromaDB (local)"
    )

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title("RAG Assistant — Live Demo")
st.caption(f"Mode: **{mode}** · Model: `{selected_model}` · k={k} · threshold={threshold:.2f}")

st.markdown(
    """
Ask a question about Central Bank of Ireland speeches.
**Left panel** answers using the raw LLM with no retrieval.
**Right panel** answers using the full RAG pipeline, grounded in the speeches corpus.
"""
)
st.divider()

# ---------------------------------------------------------------------------
# Resource loading (cached per model + retrieval parameters)
# ---------------------------------------------------------------------------


@st.cache_resource
def _get_vectorstore():
    return load_vectorstore()


@st.cache_resource(show_spinner="Building RAG chain...")
def _get_rag_chain(model: str, k: int, threshold: float):
    vs = _get_vectorstore()
    return build_rag_chain(vs, model=model, k=k, score_threshold=threshold)


@st.cache_resource(show_spinner="Connecting to Ollama...")
def _get_baseline_llm(model: str):
    return ChatOllama(model=model)


def _load_resources(model: str, k: int, threshold: float):
    try:
        chain = _get_rag_chain(model, k, threshold)
        llm = _get_baseline_llm(model)
        return chain, llm, None
    except Exception as exc:
        return None, None, str(exc)


chain, llm, load_error = _load_resources(selected_model, k, threshold)

if load_error:
    st.error(
        f"Could not initialise pipeline: {load_error}\n\n"
        "Make sure `ollama serve` is running and the ChromaDB collection is populated "
        "(`python scripts/ollama_ready.py`)."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Question input
# ---------------------------------------------------------------------------

question = st.text_input(
    "Ask a question:",
    placeholder="e.g. What is the Central Bank's view on inflation in 2023?",
)

ask_btn = st.button("Ask", type="primary", disabled=not question)

# ---------------------------------------------------------------------------
# Comparison panels
# ---------------------------------------------------------------------------

if ask_btn and question:
    col_left, col_right = st.columns(2, gap="large")

    # --- Left: Baseline (no RAG) ---
    with col_left:
        st.subheader("Without RAG")
        st.caption("Raw LLM — no retrieval, no grounding in the corpus")
        with st.spinner("Generating baseline answer..."):
            t0 = time.perf_counter()
            baseline_response = llm.invoke([HumanMessage(content=question)])
            baseline_elapsed = time.perf_counter() - t0
        st.write(baseline_response.content)
        st.metric("Response time", f"{baseline_elapsed:.2f} s")

    # --- Right: RAG ---
    with col_right:
        st.subheader("With RAG")
        st.caption("Retrieval-augmented — grounded in CBI speeches")
        with st.spinner("Retrieving chunks and generating answer..."):
            t0 = time.perf_counter()
            result = ask(
                chain,
                question,
                fallback_llm=llm if use_fallback else None,
            )
            rag_elapsed = time.perf_counter() - t0

        rag_answer = result["answer"]
        conf = result["confidence"]
        sources = result["sources"]
        scores = result["scores"]

        st.write(rag_answer)

        # Timing
        st.metric("Response time", f"{rag_elapsed:.2f} s",
                  delta=f"{rag_elapsed - baseline_elapsed:+.2f} s vs baseline",
                  delta_color="inverse")

        # Confidence / fallback status
        if conf["used_fallback"]:
            st.warning("Fallback active — answer is from general knowledge, not the corpus.")
        elif conf["is_uncertain"]:
            st.error(f"No answer found in corpus ({conf['n_sources']} chunks retrieved).")
        else:
            level = conf["confidence_level"]
            colour = {"High": "✅", "Medium": "🟡", "Low": "🟠", "Very Low": "🔴"}.get(level, "❓")
            st.success(
                f"{colour} **{level}** confidence · "
                f"{conf['n_sources']} chunks · "
                f"{conf['unique_documents']} document(s)"
            )

        # Metadata block
        meta_rows = [
            ("Retrieved chunks", conf["n_sources"]),
            ("Unique documents", conf["unique_documents"]),
            ("Confidence level", conf["confidence_level"]),
        ]
        if show_scores and conf["top_score"] is not None:
            meta_rows += [
                ("Top similarity score", f"{conf['top_score']:.3f}"),
                ("Avg similarity score", f"{conf['avg_score']:.3f}"),
            ]

        if show_scores or mode == "Debug":
            with st.expander("Retrieval metadata"):
                for label, val in meta_rows:
                    st.markdown(f"**{label}:** {val}")

        # Sources
        if sources:
            with st.expander(f"Retrieved sources ({len(sources)} chunks)"):
                for i, (src, score) in enumerate(zip(sources, scores or [None] * len(sources)), 1):
                    title = src.get("title") or src.get("source", "unknown")
                    speaker = src.get("speaker", "")
                    date = src.get("date", "")
                    meta = " · ".join(p for p in [speaker, date] if p)
                    score_str = f" — score {score:.3f}" if score is not None and show_scores else ""
                    st.markdown(f"**{i}.** {title}{score_str}")
                    if meta:
                        st.caption(meta)
                    if show_raw_chunks and i <= len(result["sources"]):
                        # Retrieve raw chunk text from chain result (not stored in sources)
                        pass  # raw text would need source_docs reference; shown via scores panel

st.divider()
st.caption(
    "Project: *Building a RAG System I Can Explain* · "
    "Alan McDonagh (A00052777) · TU Dublin CPD Foundations of AI"
)
