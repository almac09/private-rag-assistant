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
7. [Reflections](#7-reflections)
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
| Live demo app | `app.py` -- Streamlit app: baseline LLM vs RAG side-by-side |
| Test suite | `tests/` -- 118 passing unit tests, all CI-safe |
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

**Streamlit demo app** (`app.py`): A live interactive comparison tool built with Streamlit.
Users type a question and see two answers side by side -- the raw LLM response (no retrieval)
and the RAG pipeline response (retrieved from the CBI speeches corpus), with the confidence
signal and source metadata displayed beneath. Run with `uv run streamlit run app.py`.

- Issue [#43](https://github.com/almac09/private-rag-assistant/issues/43) -- Streamlit app
- `quarto/demo_app.qmd` -- demo page with code walkthrough and comparison output

---

## 3 Evaluation

### 3.1 Test Suite

99 unit tests across 6 test files, all passing in CI on Python 3.11 (Ubuntu):

| File | Tests | What it covers |
|------|-------|---------------|
| `test_ingest.py` | 20 | CSV loading, chunking, ChromaDB embedding |
| `test_query.py` | 39 | RAG chain construction, ask(), confidence_signal(), score filtering, fallback |
| `test_evaluate.py` | 15 | test_questions.json schema, load_test_questions(), save_results() |
| `test_logger.py` | 18 | log_query(), is_failed_query(), load_log(), replay_failed() |
| `test_app.py` | 27 | AppTest render/interaction, app structure, modes, timing, fallback toggle |
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
4. A Streamlit live demo app comparing baseline vs RAG side by side (`app.py`)
5. A structured meta-AI governance workflow (see Appendix I)
6. Full Sphinx HTML documentation and Quarto published site

**Open items (tracked as GitHub issues):**

| Issue | Description | Priority |
|-------|-------------|----------|
| [#44](https://github.com/almac09/private-rag-assistant/issues/44) | Run RAGAS evaluation and fill in §3.3 scores | Must-do — §3.3 has placeholders |
| [#55](https://github.com/almac09/private-rag-assistant/issues/55) | LLM response latency — implement streaming output and num_predict cap | Must-do — 20–45s response time is unusable in demo |
| [#57](https://github.com/almac09/private-rag-assistant/issues/57) | Quarto site final polish — staleness review, update all pages for new features | Must-do — blocks submission |
| [#45](https://github.com/almac09/private-rag-assistant/issues/45) | Document chunk quality checks in §2.1 | Nice-to-have |
| [#52](https://github.com/almac09/private-rag-assistant/issues/52) | Retrieval quality improvements — hybrid search, reranking, query rewriting | Future work |

---

## 7 Reflections

### 7.1 What I Learned About RAG Systems

Building this system from scratch — rather than using a pre-packaged API — forced genuine understanding
of each component. The key insight was that RAG is not one thing: it is a chain of four decisions, each
with its own failure mode.

**Retrieval quality determines answer quality.** No prompt engineering can compensate for retrieving the
wrong chunks. The most instructive failure was discovering that LangChain's `as_retriever()` interface
does not expose similarity scores — the pipeline was retrieving chunks but had no way to know how
relevant they were. Switching to `similarity_search_with_relevance_scores()` and introducing a
`score_threshold` parameter made relevance visible and filterable. This single change made the
confidence signal meaningful.

**The knowledge base choice matters more than the model.** The original plan used the course slide PDFs.
The lecturer's advice to switch to the CBI speeches corpus turned out to be correct for a non-obvious
reason: slides are written to accompany speech, not to stand alone. Extracted text from a slide deck
like "Slide 14: Key risks — see previous slide" contains almost no retrievable information. Full-text
speeches, by contrast, contain complete arguments and evidence. Garbage in, garbage out applies at the
corpus level before a single embedding is computed.

**Confidence scoring is harder than it sounds.** The heuristic confidence signal (High / Medium / Low /
Very Low based on top similarity score) is useful but not honest. A model can return a confident-sounding
answer that scores 0.9 on cosine similarity but is still wrong, because cosine similarity measures
vector proximity, not factual accuracy. RAGAS faithfulness — which cross-checks the answer against the
retrieved chunks — is the closer approximation to real quality. The heuristic is fast and useful for
flagging edge cases; it is not a substitute for human evaluation.

**Local LLMs are slow.** A 7B-parameter model on CPU takes 20–45 seconds per response. This is not a
problem during development (tests are all mocked) but is very noticeable in the live demo. The practical
fix is model selection (1B models respond in 2–5 seconds) or streaming output (tokens appear as they
generate, so the wait feels shorter even if wall-clock time is unchanged). Issue
[#55](https://github.com/almac09/private-rag-assistant/issues/55) tracks this.

**The abstain phrase is the most important design decision in the prompt.** Grounding the model to the
corpus with `"If the answer is not in the context, reply: '...'"` and then detecting that exact phrase
downstream creates a single, testable control point. Every confidence check, fallback trigger, and
query log failure flag depends on this one sentinel. If the phrase changes, everything downstream
breaks — which is why it lives in one constant (`ABSTAIN_PHRASE`) imported everywhere rather than
duplicated as a string literal.

### 7.2 What I Learned About AI-Assisted Development

This project had a second deliverable: using Claude Code as a junior developer under human direction,
and documenting the experience for an AI tools appendix.

**The discipline of the issue workflow was the most valuable constraint.** The requirement to create a
GitHub issue before writing code, write tests before merging, and close with an attribution comment
sounds bureaucratic. In practice it produced a searchable audit trail where every design decision can
be reconstructed from the commit history. This is the kind of traceability a regulator or senior
reviewer would ask for — and it would have been difficult to retrofit if left to the end.

**AI assistance accelerates the mechanical parts and exposes the thinking parts.** Writing boilerplate
tests, setting up Sphinx config, and drafting GitHub issue bodies took seconds. The parts that required
human judgment — which knowledge base to use, how to structure the confidence signal, whether to add a
feature or defer it — still took the same amount of thinking. The net effect was that more time was
spent on design decisions and less on typing. For a solo project with a deadline, that is a meaningful
shift.

**Mocking is not as simple as it looks.** The most technically interesting debugging session in this
project was discovering that LangChain's LCEL pipeline calls `llm(input)` rather than `llm.invoke(input)`
when the model is not a `Runnable` subclass (it wraps the object in a `RunnableLambda`). This meant
that `mock_llm.invoke.return_value` was never reached, and the mock returned a raw `MagicMock` object
rather than an `AIMessage`. The fix (`mock_llm.return_value.return_value = AIMessage(...)`) required
understanding the LangChain execution model, not just the test framework. No amount of prompt
engineering finds that bug — reading the source does.

**AI tools produce confident-sounding incorrect output.** A stale package name (`sphinxcontrib-docxbuilder`,
which does not exist on PyPI) was added to `pyproject.toml` without error. A Windows-only em-dash
character in a docstring caused a `SyntaxError` only on the Ubuntu CI runner. Both were caught by CI,
not by review. The lesson: CI is not optional when AI writes code.

### 7.3 What I Would Do Differently

**Start evaluation earlier.** The RAGAS harness was built before the corpus was loaded, which is the
right engineering order. But actually *running* the evaluation — generating real scores — was deferred.
The result is that §3.3 of this report has placeholders rather than numbers. For a graded project, that
is a gap. If I started again, I would run a small evaluation (5 questions, any reasonable model) in
Phase 2 to establish a baseline, then improve it iteratively.

**Choose the model first, then tune the rest.** The demo app was built assuming `llama3.2` (7B), then
discovered to be unusably slow on the demo machine. Model selection should be the first experiment, not
something discovered at the demo stage.

**Chunk size deserves its own experiment.** `chunk_size=1000, chunk_overlap=100` was the default choice
from LangChain examples. Whether that is the right setting for CBI speeches — which are long-form,
argument-driven documents — was never tested. A short ablation (chunk sizes 500, 750, 1000, 1500 with
fixed evaluation questions) would have produced a defensible choice rather than an inherited default.

### 7.4 Connection to Professional Practice

My day job involves constructing and communicating probabilistic models to regulators and boards who
did not build them. The RAG system surfaces a parallel problem: *how do you know the answer is right,
and how do you communicate your uncertainty?*

The confidence signal — even as a heuristic — demonstrates that question explicitly. An answer labelled
"High confidence · 4 chunks · 2 documents" communicates something different from "Very Low · 0 chunks
retrieved". A model that only ever returns confident-sounding answers with no indication of retrieval
quality is not suitable for professional use, regardless of how polished the interface looks.

The same principle applies to actuarial models: a number without an indication of its uncertainty and
the assumptions behind it is incomplete. The ABSTAIN_PHRASE, the fallback mechanism, and the confidence
levels in this project are the RAG equivalent of confidence intervals and model limitations sections —
not features added for the demo, but part of what it means for the system to be honest.

---



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
