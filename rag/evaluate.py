"""RAGAS evaluation harness: score RAG outputs for faithfulness and answer relevancy."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from rag.query import ask


def load_test_questions(path: str | Path = "tests/test_questions.json") -> list[dict]:
    """Load the gold-standard question set.

    Each entry must have keys ``question`` and ``ground_truth``.
    Entries with ``in_corpus: false`` are included for abstention testing.

    Args:
        path: Path to the JSON file.

    Returns:
        List of question dicts.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If any entry is missing required keys.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Test questions file not found: {path}")
    with open(p) as f:
        questions = json.load(f)
    required = {"id", "question", "ground_truth"}
    for i, q in enumerate(questions):
        missing = required - q.keys()
        if missing:
            raise ValueError(f"Question at index {i} missing keys: {missing}")
    return questions


def run_evaluation(
    chain,
    questions: list[dict],
    phase: str = "unknown",
) -> dict:
    """Run all test questions through the chain and score with RAGAS.

    Requires Ollama to be running and the vectorstore to be populated.
    For each question, calls ``chain.invoke()`` to get the answer and
    retrieved contexts, then evaluates with four RAGAS metrics.

    Args:
        chain: LangChain RAG chain from :func:`rag.query.build_rag_chain`.
        questions: List of dicts from :func:`load_test_questions`.
        phase: Label for this evaluation run (e.g. ``"phase_2"``).

    Returns:
        Dict with keys ``phase``, ``timestamp``, ``questions`` (per-question
        results), and ``scores`` (aggregate RAGAS metrics).
    """
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
    except ImportError as e:
        raise ImportError(
            "RAGAS and/or datasets not installed. Run: uv add ragas datasets"
        ) from e

    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    per_question = []

    for q in questions:
        result = ask(chain, q["question"])
        rows["question"].append(q["question"])
        rows["answer"].append(result["answer"])
        rows["contexts"].append([s.get("page_content", "") for s in result["sources"]])
        rows["ground_truth"].append(q["ground_truth"])
        per_question.append(
            {
                "id": q["id"],
                "question": q["question"],
                "answer": result["answer"],
                "sources": result["sources"],
                "ground_truth": q["ground_truth"],
                "tags": q.get("tags", []),
            }
        )

    dataset = Dataset.from_dict(rows)
    scores = ragas_evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    return {
        "phase": phase,
        "timestamp": datetime.utcnow().isoformat(),
        "n_questions": len(questions),
        "questions": per_question,
        "scores": {
            "faithfulness": float(scores["faithfulness"]),
            "answer_relevancy": float(scores["answer_relevancy"]),
            "context_precision": float(scores["context_precision"]),
            "context_recall": float(scores["context_recall"]),
        },
    }


def save_results(results: dict, path: str | Path | None = None) -> Path:
    """Persist evaluation results to JSON.

    Args:
        results: Results dict from :func:`run_evaluation`.
        path: Output path. Defaults to
            ``tests/results/phase_<phase>_<YYYYMMDD>.json``.

    Returns:
        The path the file was written to.
    """
    if path is None:
        date_str = datetime.utcnow().strftime("%Y%m%d")
        phase = results.get("phase", "unknown").replace(" ", "_")
        path = Path("tests") / "results" / f"phase_{phase}_{date_str}.json"
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    return out
