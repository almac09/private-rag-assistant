# Private RAG Assistant — Project Brief
> Paste this file at the start of every Claude session to restore full context.

---

## Who I Am
- **Name:** Alan McDonagh (A00052777)
- **Course:** CPD Certificate in Foundations of AI — Krisolis / TU Dublin
- **Project:** Capstone Project (70% of final grade)
- **GitHub:** https://github.com/almac09/private-rag-assistant
- **Local repo:** C:\Users\mcdon\GitHub\private-rag-assistant

---

## What This Project Is

A **Retrieval-Augmented Generation (RAG) chatbot** grounded in the **Solvency II regulatory corpus** (EIOPA guidelines, Central Bank of Ireland circulars — all publicly available).

### The Real-World Backstory
At Milliman (previous role), Alan bootstrapped a RAG system for the Dublin office by piggybacking a Seattle IT proof-of-concept. It used an air-gapped internal LLM pointed at Solvency II docs on SharePoint. It got presented at the European conference, Dublin was recognised as trailblazers. A senior partner wanted to extend it to M&A due diligence. But it had no feedback loop, no confidence scoring, and required the original builder to fix failures. This project fixes all of that.

### Core Constraint
**No data must leave the machine.** The architecture must work identically at home (personal laptop) and at work (regulated environment). This means:
- Primary model: **Ollama** (fully local, no API, no data egress)
- Fallbacks: OpenAI → Anthropic → Gemini (skipped gracefully if no key in `.env`)
- Document source: **public PDFs only** (EIOPA, CBI) — no confidential data ever in the pipeline

---

## Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Repo setup, hello world per provider | 🔲 Not started |
| 1 | Document ingestion (chunk + embed Solvency II PDFs) | 🔲 Not started |
| 2 | RAG query pipeline (retrieve + generate + cite source) | 🔲 Not started |
| 3 | Evaluation harness (RAGAS + deterministic seed test set) | 🔲 Not started |
| 4 | Feedback loop + confidence scoring | 🔲 Not started |
| 5 | Enterprise deployment plan (report section) | 🔲 Not started |

---

## Repo Structure

```
private-rag-assistant/
│
├── PROJECT_BRIEF.md          ← this file — paste to Claude each session
├── README.md
├── .gitignore
├── .env.example              ← key template (copy to .env, never commit .env)
├── requirements.txt
│
├── docs/                     ← Solvency II PDFs (git-ignored, downloaded locally)
│
├── 00_hello_world.py         ← test each LLM provider, graceful fallback
├── 01_ingest.py              ← download + chunk + embed docs → vector store
├── 02_query.py               ← RAG query: retrieve chunks → generate answer + cite
├── 03_evaluate.py            ← RAGAS scoring + seed-based failure replay
│
├── test_questions.json       ← 20-question gold standard test set
│
└── notebooks/
    └── exploration.ipynb
```

---

## Tech Stack

| Component | Tool | Reason |
|-----------|------|--------|
| Primary LLM | Ollama (llama3 or mistral) | Fully local, no data egress |
| Fallback LLMs | OpenAI / Anthropic / Gemini | Dev convenience, skipped if no key |
| Embeddings | Ollama (`nomic-embed-text`) | Local, no API needed |
| Vector store | ChromaDB | Runs locally, no server needed |
| RAG framework | LangChain | Industry standard, well documented |
| Evaluation | RAGAS | Purpose-built RAG evaluation |
| Key management | python-dotenv + `.env` | Keys never hardcoded |

---

## Report Status

| Section | Status | Notes |
|---------|--------|-------|
| Project Overview | ✅ Written (v003) | Polished from stream-of-consciousness notes |
| Project Objective | ✅ Written (v003) | Clear 3-component architecture stated |
| Outline of Solution | ✅ Written (v003) | Ingestion → RAG → RAGAS eval |
| Solution Development | 🔲 Not started | Write after Phase 1-2 complete |
| Evaluation | 🔲 Not started | Write after Phase 3 complete |
| Legal & Ethical | 🔲 Not started | GDPR, AI Act, public data rationale |
| Conclusions | 🔲 Not started | Write last |
| AI Tools Appendix | 🔲 Not started | Log as we go |

**Report doc:** `%USERPROFILE%\OneDrive\Projects\01 - TUD GenAI\0B. Assessment Details\02. Capstone Project Details\Workings\Capstone_Project_Report_Alan_McDonagh_v003.docx`

---

## Marking Scheme (for Claude to keep us honest)

| Component | Weight |
|-----------|--------|
| Project Initiation Document | 20% |
| Solution Development | 50% |
| Legal & Ethical Considerations | 10% |
| Solution Report | 20% |

---

## Key Decisions Made

- **Public docs only** → sidesteps all data governance issues at work
- **Ollama first** → architecture portable from laptop to enterprise
- **RAGAS evaluation** → professional-grade scoring without human graders
- **Deterministic seed test harness** → replay failures across versions
- **English:** Report written at "polished professional" level (Claude drafts, Alan reviews)

---

## How to Use This Brief

1. Start a new Claude session at claude.ai
2. Paste the contents of this file
3. Say what you want to work on today
4. Claude will pick up exactly where we left off

*Last updated: 2026-04-24 — Sections 1.1 and 1.2 written, repo structure agreed, tech stack decided*
