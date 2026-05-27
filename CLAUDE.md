# CLAUDE.md — RAG Assistant Project

This file gives Claude Code the context it needs to work effectively in this repo.

## What this project is

A local RAG (Retrieval-Augmented Generation) pipeline built as a TU Dublin CPD capstone project.
Author: Alan McDonagh (A00052777). The goal is understanding — building something small enough to
explain and evaluate honestly, not a production system.

## Stack

| Layer | Tool |
|-------|------|
| RAG framework | LangChain |
| Vector store | ChromaDB (local) |
| LLM | Ollama (local, primary) → OpenAI → Anthropic → Gemini (fallback chain) |
| Embeddings | Ollama nomic-embed-text |
| Evaluation | RAGAS + pytest |
| Docs | Sphinx (HTML + docx) + Quarto (notebooks/reports) |
| Package manager | UV |
| Python | ≥3.11 |

## Directory layout

```
private-rag-assistant/
├── rag/                    # Core Python module (importable, Sphinx autodoc target)
│   ├── __init__.py
│   ├── ingest.py           # PDF loading, chunking, embedding → ChromaDB
│   ├── query.py            # Retrieval + generation chain
│   └── evaluate.py         # RAGAS evaluation harness
├── notebooks/              # Jupyter exploration notebooks (one per phase)
├── tests/                  # pytest suite; test_questions.json is the gold standard
├── docs/
│   ├── _source/            # Sphinx source (Markdown via MyST)
│   └── _build/             # Generated output (gitignored)
├── quarto/                 # Quarto presentation and site sources
├── data/                   # Local knowledge base PDFs (gitignored — never commit)
├── 00_hello_world.py       # LLM provider smoke test
├── pyproject.toml          # UV-managed dependencies
└── .env                    # API keys (gitignored — never commit)
```

## Development phases

| Phase | Goal | Status |
|-------|------|--------|
| 0 | Repo setup + hello world per provider | Complete |
| 1 | Document ingestion (chunk + embed course PDFs) | Not started |
| 2 | RAG query pipeline (retrieve + generate + cite) | Not started |
| 3 | Evaluation harness (RAGAS + deterministic test set) | Not started |
| 4 | Feedback loop + confidence scoring | Not started |
| 5 | Enterprise deployment plan | Not started |

## Key constraints

- **Nothing leaves the laptop.** Ollama is the primary LLM; cloud providers only if API key present.
- **Knowledge base = CBI speeches corpus** (lecturer advised against course slide PDFs — they are slide decks designed to accompany spoken delivery; extracted text is too thin for good retrieval. Use the Central Bank of Ireland speeches dataset supplied with the course instead).
- **Scope is deliberately small.** Don't add features beyond what the current phase needs.

## Meta-AI tracking strand

This project has a second deliverable beyond the RAG pipeline: a **meta-analysis of AI-assisted project management**. GitHub issues, milestones, branches, and PRs are created via Claude Code and GitHub Copilot under human direction, mirroring professional engineering governance. Keep commit messages and issue bodies high-quality — they are evidence for the meta-analysis and the AI Tools appendix.

### Separating AI commits from human commits

Claude Code appends `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>` to every commit it creates. To filter:

```powershell
# Commits with AI involvement
git log --grep="Co-Authored-By: Claude"

# Commits by Alan only
git log --invert-grep --grep="Co-Authored-By"
```

The co-author line also makes Claude appear as a named contributor in GitHub's contributor graph — useful evidence for the meta-analysis.

## Running things

```powershell
# Install / sync dependencies
uv sync

# Run smoke test (checks each LLM provider in turn)
python 00_hello_world.py

# Build Sphinx docs (HTML)
sphinx-build -b html docs\_source docs\_build

# Build Sphinx docs (Word)
sphinx-build -b docx docs\_source docs\_build

# Run test suite
pytest tests/
```

## Code style

- **Black** for formatting (line length 100). Run via VSCode task "08. Black Format Codebase".
- **Ruff** for linting.
- Docstrings on every public function — Sphinx autodoc reads them.
- No comments explaining *what* the code does; only *why* when non-obvious.

## Session logs

Save Claude Code session logs with `python save_session.py`. They go in `log_AI/` and feed the
AI Tools appendix of the capstone report.
