# Evaluation Results

RAGAS evaluation results are published here per phase as evaluation runs complete.

Each phase table shows:

| Column | Description |
|--------|-------------|
| Question | The test question from the gold-standard set |
| Retrieved chunks | Top-k chunks returned by the retriever |
| Generated answer | The LLM's answer given those chunks |
| Faithfulness | RAGAS score: is the answer grounded in the context? (0–1) |
| Answer relevancy | RAGAS score: does the answer address the question? (0–1) |

Results are written to `tests/evaluation_results.json` when `pytest tests/` runs.

## Phase 1 — Baseline (no documents)

*Results pending.*

## Phase 2 — Single document

*Results pending.*

## Phase 3 — Expanded knowledge base

*Results pending.*

## Phase 4 — Full evaluation

*Results pending.*
