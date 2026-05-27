# Building a RAG System I Can Explain — Completion Report

**CPD Certificate in the Foundations of AI — Project Report**

**Student Name:** Alan McDonagh
**Student Number:** A00052777

*This document is the completion report. The original project proposal is in `report.md`.*

---

## Table of Contents

1. [What Was Built](#1-what-was-built)
2. [Meeting the Project Criteria](#2-meeting-the-project-criteria)
   - 2.1 [Data](#21-data)
   - 2.2 [Modelling](#22-modelling)
   - 2.3 [Deployment](#23-deployment)
3. [Evaluation](#3-evaluation)
   - 3.1 [Test Suite](#31-test-suite)
   - 3.2 [RAGAS Evaluation Design](#32-ragas-evaluation-design)
   - 3.3 [RAGAS Results](#33-ragas-results)
4. [Phase 4 — Feedback Loop](#4-phase-4--feedback-loop)
5. [Legal and Ethical Considerations](#5-legal-and-ethical-considerations)
6. [Conclusions](#6-conclusions)
- [Appendix I: Statement on Use of AI Tools (expanded)](#appendix-i-statement-on-use-of-ai-tools-expanded)

---

## 1 What Was Built

A local Retrieval-Augmented Generation (RAG) pipeline in Python, running entirely on a personal laptop.
The pipeline loads a corpus of Central Bank of Ireland speeches, answers questions from that corpus, and
evaluates its own performance using RAGAS metrics. Nothing is sent to external servers.

**Repository:** https://github.com/almac09/private-rag-assistant

**Published documentation site:** https://almac09.github.io/private-rag-assistant/

| Component | Implementation |
|-----------|---------------|
| Document ingestion | `rag/ingest.py` — CSV loading, chunking, ChromaDB embedding |
| Query pipeline | `rag/query.py` -- retrieval chain + generation + confidence signal |
| Evaluation harness | `rag/evaluate.py` -- RAGAS metrics over 20-question test set |
| Query logging | `rag/logger.py` -- JSONL log with failed-query flagging and replay |
| Test suite | `tests/` -- 79 passing unit tests, all CI-safe |
| Documentation | Sphinx HTML + Quarto site + Quarto RevealJS presentation |

---

## 2 Meeting the Project Criteria

### 2.1 Data

**Original plan (report.md §2.1):** Load and chunk the course PDFs into a vector database.

**What happened:** The lecturer advised against using the course slide PDFs (Appendix II of the proposal),
noting they are slide decks designed to accompany spoken delivery -- extracted text is too thin for good
retrieval. The knowledge base was switched to **the Central Bank of Ireland (CBI) speeches dataset**
supplied with the course. This is a CSV of full-text speeches covering monetary policy, financial
regulation, and economic outlook.

**Evidence of implementation:**

- `rag/ingest.py` -- `load_speeches_csv()`, `chunk_documents()`, `ingest_to_chromadb()`
- `quarto/ingestion.qmd` -- live demo of the three-step pipeline with synthetic data
- Issue [#22](https://github.com/almac09/private-rag-assistant/issues/22) -- knowledge base pivot decision
- Issues [#1-#4](https://github.com/almac09/private-rag-assistant/issues/1) -- ingestion pipeline

Chunk settings: `chunk_size=1000`, `chunk_overlap=100` (configurable). Embedding model:
`nomic-embed-text` via Ollama (768-dimensional vectors).

### 2.2 Modelling

**Original plan (report.md §2.2):** LangChain retrieval + Ollama generation; retrieve 3-5 chunks per query.

**What was built:**

```
question
   -> OllamaEmbeddings (nomic-embed-text) -> ChromaDB similarity search (k=4)
   -> top-4 chunks + question -> ChatPromptTemplate
   -> ChatOllama (llama3.2) -> answer string
```

The chain is implemented using LangChain LCEL (`RunnableParallel`, `RunnablePassthrough`, `.assign()`),
which returns both the generated answer and the retrieved source documents in a single pass.

The prompt template constrains the model to the retrieved context only. If the context is empty or
off-topic, the model returns a fixed sentinel phrase (`"I don't know based on the available speeches."`)
which the confidence signal and query logger both detect.

**Evidence:**

- `rag/query.py` -- `build_rag_chain()`, `ask()`, `confidence_signal()`
- `quarto/query.qmd` -- live demo of chain construction (mocked, no Ollama needed)
- `quarto/baseline_comparison.qmd` -- comparison of baseline (no retrieval) vs RAG answers
- Issues [#5-#8](https://github.com/almac09/private-rag-assistant/issues/5)

### 2.3 Deployment

**Original plan (report.md §2.3):** Local, personal laptop, Jupyter notebooks.

**What was built:** Local Python package (`rag/`) importable by notebooks and scripts.
Ollama runs as a local server (`ollama serve`). ChromaDB persists to `chroma_db/` on disk.
Nothing is sent to external APIs unless a fallback cloud provider is configured in `.env`.

The `scripts/ollama_ready.py` pre-flight script checks that Ollama is running, the required models are
pulled, and the ChromaDB collection is populated before any query session begins.

---

## 3 Evaluation

### 3.1 Test Suite

79 unit tests across 5 test files, all passing in CI on Python 3.11 (Ubuntu):

| File | Tests | What it covers |
|------|-------|---------------|
| `test_ingest.py` | 20 | CSV loading, chunking, ChromaDB embedding |
| `test_query.py` | 21 | RAG chain construction, ask(), confidence_signal() |
| `test_evaluate.py` | 15 | test_questions.json schema, load_test_questions(), save_results() |
| `test_logger.py` | 18 | log_query(), is_failed_query(), load_log(), replay_failed() |
| `test_setup.py` | 9 | dependency imports, Ollama service (local only) |

All tests that require Ollama or real data are marked `@ollama_required` or `@real_data` and
skip automatically in CI.

### 3.2 RAGAS Evaluation Design

**Original plan (report.md §3.1):** Score outputs against a structured test set using RAGAS metrics.

**What was built:**

- `tests/test_questions.json` -- 20 questions with ground-truth answers drawn from the CBI speeches corpus
  - Fields: `id`, `question`, `ground_truth`, `tags`, `in_corpus`
  - Tags: factual, analytical, temporal, comparative, mandate, rates, instruments
  - q018 is a deliberate abstention question (`in_corpus: false`) to test refusal behaviour
- `rag/evaluate.py` -- `run_evaluation()` calls `ask()` for each question and scores with RAGAS

Four RAGAS metrics:

| Metric | What it measures |
|--------|-----------------|
| Faithfulness | Is the answer grounded in the retrieved chunks? |
| Answer Relevancy | Does the answer address the question asked? |
| Context Precision | Are the retrieved chunks relevant to the question? |
| Context Recall | Did retrieval find all chunks needed to answer? |

The evaluation is wrapped in `rag/evaluate.py:run_evaluation()` and results are saved to
`tests/results/phase_<phase>_<date>.json`.

### 3.3 RAGAS Results

*To be completed once the full CBI speeches corpus is loaded and Ollama is running.*

The evaluation run requires:

```powershell
# 1. Start Ollama
ollama serve

# 2. Confirm corpus is loaded (or load it)
python scripts/ollama_ready.py

# 3. Run the evaluation
python -c "
from rag.query import load_vectorstore, build_rag_chain
from rag.evaluate import load_test_questions, run_evaluation, save_results

vs = load_vectorstore()
chain = build_rag_chain(vs)
questions = load_test_questions()
results = run_evaluation(chain, questions, phase='phase_3')
path = save_results(results)
print(f'Results saved to {path}')
print(results['scores'])
"
```

Results table (fill in after running):

| Metric | Score |
|--------|-------|
| Faithfulness | *TBD* |
| Answer Relevancy | *TBD* |
| Context Precision | *TBD* |
| Context Recall | *TBD* |

The `quarto/evaluation.qmd` page reads results automatically from `tests/results/*.json` and renders
the table above once the file is present.

---

## 4 Phase 4 -- Feedback Loop

The original proposal (report.md §5, future work bullets) listed two items that have since been
implemented:

**Query logging** (`rag/logger.py`):

- `log_query()` appends every query to a JSONL file with fields: timestamp, question, answer, sources, failed
- `is_failed_query()` detects the abstain sentinel automatically
- `replay_failed()` returns only entries where the model could not answer, for re-testing after pipeline changes

**Confidence signal** (`rag/query.py`):

- Every `ask()` call now returns a `"confidence"` dict alongside the answer
- Fields: `is_uncertain` (bool), `n_sources` (int), `has_sources` (bool), `unique_documents` (int)
- No extra LLM calls; heuristic-based

Both are demonstrated in `quarto/logging.qmd` and `quarto/confidence.qmd` respectively.

---

## 5 Legal and Ethical Considerations

All points from the proposal (report.md §4) stand:

- Knowledge base: CBI speeches, publicly available. No personal data, no confidential information.
- LLM: Ollama (local). Nothing sent to external servers unless a cloud fallback key is configured.
- Known limitations: accuracy risk, transparency requirement, KB completeness. All documented.

---

## 6 Conclusions

The project delivered:

1. A working local RAG pipeline over the CBI speeches corpus
2. A 20-question RAGAS evaluation harness, ready to run once corpus is loaded
3. Phase 4 feedback-loop features (logging + confidence signal)
4. A structured meta-AI governance workflow (see Appendix I)
5. Full Sphinx HTML documentation and Quarto published site

What remains: running the RAGAS evaluation with the real corpus loaded and filling in the scores
in §3.3 above.

---

## Appendix I: Statement on Use of AI Tools (expanded)

### Tools used

| Tool | Role |
|------|------|
| Claude Code (Anthropic) | Primary coding assistant -- module implementation, test generation, Quarto pages, GitHub issue and PR management |
| GitHub Copilot | Inline code suggestions in VS Code |
| Claude (chat) | Report drafting and prose editing |

### How AI was used for project governance (the meta-AI strand)

Every issue in this repository was created, described, and closed using a consistent workflow:

1. **Issue creation** -- Claude Code drafted the issue title, body, and acceptance criteria. I reviewed and approved before posting.
2. **Implementation** -- Claude Code wrote the Python modules, tests, and Quarto pages. I reviewed every diff.
3. **PR creation** -- Pull requests opened by Claude Code with structured summaries and test plans.
4. **Closing comment** -- Every issue was closed with an attribution comment: `Generated by Claude Sonnet 4.6 -- actioned on Alan McDonagh's approval`
5. **Merge** -- I performed every merge, under my GitHub account.

### Audit trail

Claude Code appends `Co-Authored-By: Claude Sonnet 4.6` to every commit it creates:

| Metric | Count |
|--------|-------|
| Total commits | 60 |
| AI-involved (Co-Authored-By: Claude) | 29 |
| Human-only | 31 |
| Closed issues | 22 |
| Merged pull requests | 17 |

```bash
# Filter to AI-involved commits
git log --grep="Co-Authored-By: Claude" --oneline

# Filter to human-only commits
git log --invert-grep --grep="Co-Authored-By" --oneline
```

### What worked well

**Structured issue descriptions.** Claude Code consistently produced issue bodies with clear acceptance
criteria, making it easy to know when work was done.

**Mandatory closing comment workflow.** The attribution comment on every issue provides a searchable
audit trail: what was built, which tests cover it, which Quarto page demonstrates it.

**CI as a gate.** GitHub Actions CI + branch protection caught regressions that had passed local testing.

### What did not work as well

**Package research.** Claude Code added a package to `pyproject.toml` (`sphinxcontrib-docxbuilder`)
that does not exist on PyPI. Training-data knowledge of niche packages can be stale.

**File encoding on Windows.** Em-dash characters (U+2014) in Python docstrings triggered `SyntaxError`
on the Ubuntu CI runner. A follow-up fix was needed.

**Merge conflicts.** Parallel branches modifying the same file (`rag/query.py`) left concatenated
duplicate function bodies after automatic GitHub merge. Required manual resolution.

### Reflection for professional carry-over

The "AI raises issues, human approves and merges" pattern mirrors delegating to a junior developer.
The key discipline: the human remains the decision point at every merge. The audit trail produced --
timestamped commits, attribution comments, CI results, structured PR descriptions -- is the kind of
documentation a regulator or peer reviewer would expect when asking "who did what and when, and how
do you know it was tested?".

Session logs are in `log_AI/` (saved using `python save_session.py` after each Claude Code session).
For the full GitHub workflow, see the published site: https://almac09.github.io/private-rag-assistant/github_workflow.html
