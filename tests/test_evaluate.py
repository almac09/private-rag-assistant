"""
Tests for the RAGAS evaluation harness.

Unit tests mock RAGAS and the RAG chain — run in CI with no external dependencies.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

TEST_QUESTIONS_PATH = Path("tests/test_questions.json")


# ---------------------------------------------------------------------------
# test_questions.json structural checks
# ---------------------------------------------------------------------------

def test_questions_file_exists():
    """test_questions.json must exist."""
    assert TEST_QUESTIONS_PATH.exists(), "tests/test_questions.json not found"


def test_questions_is_valid_json():
    """test_questions.json must be parseable JSON."""
    with open(TEST_QUESTIONS_PATH) as f:
        data = json.load(f)
    assert isinstance(data, list)


def test_questions_minimum_count():
    """Test set must contain at least 20 questions."""
    with open(TEST_QUESTIONS_PATH) as f:
        data = json.load(f)
    assert len(data) >= 20, f"Expected >=20 questions, got {len(data)}"


def test_questions_required_keys():
    """Every question must have id, question, ground_truth, tags, in_corpus."""
    required = {"id", "question", "ground_truth", "tags", "in_corpus"}
    with open(TEST_QUESTIONS_PATH) as f:
        data = json.load(f)
    for q in data:
        missing = required - q.keys()
        assert not missing, f"Question {q.get('id', '?')} missing keys: {missing}"


def test_questions_unique_ids():
    """All question IDs must be unique."""
    with open(TEST_QUESTIONS_PATH) as f:
        data = json.load(f)
    ids = [q["id"] for q in data]
    assert len(ids) == len(set(ids)), "Duplicate question IDs found"


def test_questions_no_empty_text():
    """No question or ground_truth field should be empty."""
    with open(TEST_QUESTIONS_PATH) as f:
        data = json.load(f)
    for q in data:
        assert q["question"].strip(), f"Empty question text in {q['id']}"
        assert q["ground_truth"].strip(), f"Empty ground_truth in {q['id']}"


def test_questions_has_abstention_entries():
    """Test set must include at least one abstention question (in_corpus=false)."""
    with open(TEST_QUESTIONS_PATH) as f:
        data = json.load(f)
    abstentions = [q for q in data if not q.get("in_corpus", True)]
    assert abstentions, "No abstention questions (in_corpus=false) found in test set"


def test_questions_tag_variety():
    """Test set should use at least three different tag values."""
    with open(TEST_QUESTIONS_PATH) as f:
        data = json.load(f)
    all_tags = {tag for q in data for tag in q.get("tags", [])}
    assert len(all_tags) >= 3, f"Expected >=3 tag types, found: {all_tags}"


# ---------------------------------------------------------------------------
# load_test_questions
# ---------------------------------------------------------------------------

def test_load_test_questions_returns_list():
    """load_test_questions returns a list of dicts."""
    from rag.evaluate import load_test_questions
    questions = load_test_questions(TEST_QUESTIONS_PATH)
    assert isinstance(questions, list)
    assert all(isinstance(q, dict) for q in questions)


def test_load_test_questions_missing_file():
    """load_test_questions raises FileNotFoundError for missing path."""
    from rag.evaluate import load_test_questions
    with pytest.raises(FileNotFoundError):
        load_test_questions("nonexistent.json")


def test_load_test_questions_bad_format(tmp_path):
    """load_test_questions raises ValueError for entries missing required keys."""
    from rag.evaluate import load_test_questions
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"id": "q1", "question": "Q?"}]))
    with pytest.raises(ValueError, match="missing keys"):
        load_test_questions(bad)


# ---------------------------------------------------------------------------
# save_results
# ---------------------------------------------------------------------------

def test_save_results_writes_file(tmp_path):
    """save_results writes a JSON file to the specified path."""
    from rag.evaluate import save_results
    results = {"phase": "test", "timestamp": "2024-01-01", "scores": {}, "questions": []}
    out = save_results(results, path=tmp_path / "results.json")
    assert out.exists()
    with open(out) as f:
        loaded = json.load(f)
    assert loaded["phase"] == "test"


def test_save_results_creates_parent_dirs(tmp_path):
    """save_results creates parent directories if they don't exist."""
    from rag.evaluate import save_results
    results = {"phase": "p1", "timestamp": "2024-01-01", "scores": {}, "questions": []}
    deep_path = tmp_path / "a" / "b" / "results.json"
    save_results(results, path=deep_path)
    assert deep_path.exists()


def test_save_results_default_path_includes_phase(tmp_path, monkeypatch):
    """save_results default filename includes the phase label."""
    from rag.evaluate import save_results
    monkeypatch.chdir(tmp_path)
    results = {"phase": "phase_2", "timestamp": "2024-01-01", "scores": {}, "questions": []}
    out = save_results(results)
    assert "phase_2" in out.name


# ---------------------------------------------------------------------------
# run_evaluation (mocked)
# ---------------------------------------------------------------------------

def _make_mock_chain(answer="Test answer.", sources=None):
    if sources is None:
        sources = [{"speakers": "Lagarde", "title": "Test", "date": "2023-01-01",
                    "year": "2023", "source": "test.csv", "page_content": "ECB policy text."}]
    mock = MagicMock()
    mock.invoke.return_value = {
        "answer": answer,
        "source_docs": [
            MagicMock(metadata=s, page_content=s.get("page_content", ""))
            for s in sources
        ],
        "question": "Q?",
        "context": "context text",
    }
    return mock


def test_run_evaluation_structure():
    """run_evaluation returns dict with expected top-level keys (RAGAS mocked)."""
    from rag.evaluate import run_evaluation

    fake_scores = {
        "faithfulness": 0.85,
        "answer_relevancy": 0.90,
        "context_precision": 0.75,
        "context_recall": 0.80,
    }

    mock_ragas_result = MagicMock()
    mock_ragas_result.__getitem__ = lambda self, k: fake_scores[k]

    mock_dataset = MagicMock()
    mock_dataset_cls = MagicMock()
    mock_dataset_cls.from_dict.return_value = mock_dataset

    mock_ragas_mod = MagicMock()
    mock_ragas_mod.evaluate.return_value = mock_ragas_result
    mock_metrics = MagicMock()

    questions = [
        {"id": "q1", "question": "What is inflation?",
         "ground_truth": "Rising prices.", "tags": ["factual"], "in_corpus": True}
    ]

    mock_chain = MagicMock()

    with patch.dict("sys.modules", {
        "ragas": mock_ragas_mod,
        "ragas.metrics": mock_metrics,
        "datasets": MagicMock(Dataset=mock_dataset_cls),
    }), patch("rag.evaluate.ask") as mock_ask:
        mock_ask.return_value = {
            "answer": "Test answer.",
            "sources": [{"speakers": "Lagarde", "page_content": "text",
                         "title": "T", "date": "2023", "year": "2023", "source": "s"}],
        }
        result = run_evaluation(mock_chain, questions, phase="test")

    assert "phase" in result
    assert "timestamp" in result
    assert "scores" in result
    assert "questions" in result
    assert result["phase"] == "test"
    assert result["n_questions"] == 1
