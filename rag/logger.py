"""Query logging: append every RAG query to a JSONL file; mark failed queries for replay."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

# Sentinel phrase injected by the prompt template in rag/query.py.
_UNKNOWN_SENTINEL = "I don't know based on the available speeches."

_DEFAULT_LOG = Path("logs") / "query_log.jsonl"


def is_failed_query(answer: str) -> bool:
    """Return True when the model admitted it could not answer from context.

    Args:
        answer: The answer string returned by :func:`rag.query.ask`.

    Returns:
        ``True`` if the answer contains the "I don't know" sentinel phrase.
    """
    return _UNKNOWN_SENTINEL in answer


def log_query(
    question: str,
    answer: str,
    sources: list[dict],
    path: str | Path = _DEFAULT_LOG,
) -> Path:
    """Append one query result to a JSONL log file.

    Each line is a JSON object with keys ``timestamp``, ``question``,
    ``answer``, ``sources``, and ``failed``.  Failed queries (those where
    the model could not find an answer in context) are flagged so they can
    be replayed after pipeline improvements.

    Args:
        question: The question that was asked.
        answer: The answer returned by the RAG chain.
        sources: List of metadata dicts from retrieved chunks.
        path: Destination JSONL file.  Parent directories are created as needed.

    Returns:
        The path the entry was written to.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": question,
        "answer": answer,
        "sources": sources,
        "failed": is_failed_query(answer),
    }
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return out


def load_log(path: str | Path = _DEFAULT_LOG) -> list[dict]:
    """Read a JSONL query log back into a list of dicts.

    Args:
        path: Path to the JSONL file written by :func:`log_query`.

    Returns:
        List of query-result dicts (empty list if file does not exist).
    """
    p = Path(path)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def replay_failed(path: str | Path = _DEFAULT_LOG) -> list[dict]:
    """Return only the entries where ``failed`` is ``True``.

    Args:
        path: Path to the JSONL file.

    Returns:
        Subset of log entries where the model could not answer from context.
    """
    return [e for e in load_log(path) if e.get("failed")]
