"""Tests for rag.logger — query logging and failed-query detection."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from rag.logger import (
    is_failed_query,
    log_query,
    load_log,
    replay_failed,
    _UNKNOWN_SENTINEL,
)


# ---------------------------------------------------------------------------
# is_failed_query
# ---------------------------------------------------------------------------


def test_is_failed_query_detects_sentinel():
    assert is_failed_query(_UNKNOWN_SENTINEL) is True


def test_is_failed_query_detects_sentinel_embedded():
    assert is_failed_query(f"Sorry, {_UNKNOWN_SENTINEL}") is True


def test_is_failed_query_returns_false_for_real_answer():
    assert is_failed_query("The ECB raised rates by 25 basis points in July 2022.") is False


def test_is_failed_query_empty_string():
    assert is_failed_query("") is False


# ---------------------------------------------------------------------------
# log_query
# ---------------------------------------------------------------------------


def test_log_query_returns_path(tmp_path):
    log = tmp_path / "q.jsonl"
    result = log_query("What is QE?", "Quantitative easing is...", [], path=log)
    assert result == log


def test_log_query_writes_valid_jsonl(tmp_path):
    log = tmp_path / "q.jsonl"
    log_query("What is QE?", "Quantitative easing is...", [], path=log)
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["question"] == "What is QE?"
    assert entry["answer"] == "Quantitative easing is..."
    assert entry["failed"] is False


def test_log_query_marks_failed_true(tmp_path):
    log = tmp_path / "q.jsonl"
    log_query("Unknown?", _UNKNOWN_SENTINEL, [], path=log)
    entry = json.loads(log.read_text())
    assert entry["failed"] is True


def test_log_query_includes_timestamp(tmp_path):
    log = tmp_path / "q.jsonl"
    log_query("Q?", "A.", [], path=log)
    entry = json.loads(log.read_text())
    assert "timestamp" in entry
    assert len(entry["timestamp"]) > 10


def test_log_query_includes_sources(tmp_path):
    log = tmp_path / "q.jsonl"
    sources = [{"source": "speech_001.txt", "speaker": "Lagarde"}]
    log_query("Q?", "A.", sources, path=log)
    entry = json.loads(log.read_text())
    assert entry["sources"] == sources


def test_log_query_appends_multiple_entries(tmp_path):
    log = tmp_path / "q.jsonl"
    log_query("Q1?", "A1.", [], path=log)
    log_query("Q2?", "A2.", [], path=log)
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["question"] == "Q1?"
    assert json.loads(lines[1])["question"] == "Q2?"


def test_log_query_creates_parent_dirs(tmp_path):
    log = tmp_path / "nested" / "deep" / "q.jsonl"
    log_query("Q?", "A.", [], path=log)
    assert log.exists()


# ---------------------------------------------------------------------------
# load_log
# ---------------------------------------------------------------------------


def test_load_log_returns_list(tmp_path):
    log = tmp_path / "q.jsonl"
    log_query("Q?", "A.", [], path=log)
    entries = load_log(log)
    assert isinstance(entries, list)
    assert len(entries) == 1


def test_load_log_returns_empty_for_missing_file(tmp_path):
    assert load_log(tmp_path / "nonexistent.jsonl") == []


# ---------------------------------------------------------------------------
# replay_failed
# ---------------------------------------------------------------------------


def test_replay_failed_filters_only_failed(tmp_path):
    log = tmp_path / "q.jsonl"
    log_query("Good Q?", "Good answer.", [], path=log)
    log_query("Bad Q?", _UNKNOWN_SENTINEL, [], path=log)
    failed = replay_failed(log)
    assert len(failed) == 1
    assert failed[0]["question"] == "Bad Q?"


def test_replay_failed_empty_when_all_pass(tmp_path):
    log = tmp_path / "q.jsonl"
    log_query("Q?", "A.", [], path=log)
    assert replay_failed(log) == []


def test_replay_failed_returns_empty_for_missing_file(tmp_path):
    assert replay_failed(tmp_path / "nonexistent.jsonl") == []


# ---------------------------------------------------------------------------
# ask() integration — log_path wiring
# ---------------------------------------------------------------------------


def test_ask_logs_when_log_path_provided(tmp_path):
    """ask() calls log_query when log_path is supplied."""
    log = tmp_path / "q.jsonl"
    mock_chain = MagicMock()
    mock_doc = MagicMock()
    mock_doc.metadata = {"source": "s.txt"}
    mock_chain.invoke.return_value = {
        "answer": "ECB raised rates.",
        "source_docs": [mock_doc],
    }

    with patch("rag.logger.log_query") as mock_log:
        from rag.query import ask

        ask(mock_chain, "What did the ECB do?", log_path=log)
        mock_log.assert_called_once()
        call_kwargs = mock_log.call_args
        assert call_kwargs[0][0] == "What did the ECB do?"
        assert call_kwargs[0][1] == "ECB raised rates."


def test_ask_does_not_log_when_no_log_path():
    """ask() skips logging when log_path is None (default)."""
    mock_chain = MagicMock()
    mock_doc = MagicMock()
    mock_doc.metadata = {}
    mock_chain.invoke.return_value = {"answer": "A.", "source_docs": [mock_doc]}

    with patch("rag.logger.log_query") as mock_log:
        from rag.query import ask

        ask(mock_chain, "Q?")
        mock_log.assert_not_called()
