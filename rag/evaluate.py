"""RAGAS evaluation harness: score RAG outputs for faithfulness and answer relevancy."""

from __future__ import annotations

import json
from pathlib import Path


def load_test_questions(path: str | Path = "tests/test_questions.json") -> list[dict]:
    """Load the gold-standard question set.

    Each entry should have keys: ``question``, ``ground_truth``.

    Args:
        path: Path to the JSON file.

    Returns:
        List of question dicts.
    """
    with open(path) as f:
        return json.load(f)


def run_evaluation(chain: object, questions: list[dict]) -> dict:
    """Run all test questions through the chain and score with RAGAS.

    Args:
        chain: LangChain RAG chain from :func:`rag.query.build_rag_chain`.
        questions: List of dicts from :func:`load_test_questions`.

    Returns:
        Dict with per-question results and aggregate RAGAS scores
        (``faithfulness``, ``answer_relevancy``).
    """
    raise NotImplementedError("Phase 3")


def save_results(results: dict, path: str | Path = "tests/evaluation_results.json") -> None:
    """Persist evaluation results to JSON for Sphinx/Quarto rendering.

    Args:
        results: Results dict from :func:`run_evaluation`.
        path: Output path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
