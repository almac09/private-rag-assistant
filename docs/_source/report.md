# Building a RAG System I Can Explain

**CPD Certificate in the Foundations of AI — Project Report Document**

**Student Name:** Alan McDonagh  
**Student Number:** A00052777

---

## Table of Contents

1. [Project Overview](#1-project-overview)
   - 1.1 [Project Objective](#11-project-objective)
   - 1.2 [Outline of Solution](#12-outline-of-solution)
2. [Solution Development](#2-solution-development)
   - 2.1 [Data](#21-data)
   - 2.2 [Modelling](#22-modelling)
   - 2.3 [Deployment](#23-deployment)
3. [Evaluation](#3-evaluation)
   - 3.1 [Evaluation Design](#31-evaluation-design)
   - 3.2 [Evaluation Results](#32-evaluation-results)
   - 3.3 [Evaluation Discussion](#33-evaluation-discussion)
4. [Legal and Ethical Considerations](#4-legal-and-ethical-considerations)
5. [Conclusions and Future Work](#5-conclusions-and-future-work)
- [References](#references)
- [Appendix I: Statement on Use of AI Tools](#appendix-i-statement-on-use-of-ai-tools)

---

## 1 Project Overview

This project grew out of my previous day job as a consultant at an actuarial firm. Anyone who has tried to find a specific paragraph of Solvency II guidance under time pressure knows the drill — the text runs to thousands of pages, and the search tools we had made the job slower than it needed to be. A colleague in another office was piloting a Retrieval-Augmented Generation (RAG) setup — basically, wiring a large language model to a curated set of documents so it answers from those documents rather than from whatever it picked up in training. I put the Dublin office forward as a test site and got to see how it worked in practice.

It worked well enough to be useful, but a few things nagged. When the chatbot gave a bad answer, we had no clean way to log it, no way to replay it against a fixed version, and nothing on screen to tell the user how much to trust the response in front of them. Anything we wanted to change had to go back through the original architect, which slowed everything down. And when colleagues asked me why a particular answer was wrong, or what it would take to fix it, or what good actually looked like — I struggled to give a clear answer.

This project is my attempt to close that gap. I'm not promising to solve every one of those problems in a capstone — what I want is enough hands-on understanding to describe them clearly, test them in a controlled setting, and report back honestly on what I find. The reporting-back part matters as much to me as the build itself.

### 1.1 Project Objective

The main goal here is to build a simple, working RAG system from the PDFs and Jupyter notebooks the course provides. I'm not aiming for a production-ready tool. I want to get my hands on the pieces of a RAG pipeline, see how they fit together, find out where it tends to break, and learn how to evaluate the outputs in a structured way rather than by eye.

The second goal is to get better at explaining what a system like this can and can't do to people who don't build them — colleagues, clients, regulators. That means being able to talk about a failure without waving it away, and presenting what I find in a way that would actually be useful back at the desk if we pointed the same approach at internal or regulatory documents.

I've deliberately kept the scope small. The system starts with no documents loaded and is expected to fail on basic questions — that's the point. Watching it fail in a controlled setting, working out why, and writing it up honestly is a perfectly good outcome rather than something to paper over.

### 1.2 Outline of Solution

I'm planning to build a Python RAG pipeline using the tools the course introduces: LangChain for the retrieval and generation logic, ChromaDB as a local vector store, and an open-source language model running locally via Ollama. I'm prototyping in Jupyter notebooks — where code, outputs, and my running commentary sit side by side, which also makes the whole thing reasonably readable for someone who doesn't write Python — and as bits of it stabilise I'm pulling them out into a proper Python module that the notebooks then call back into. That way the messy exploration stays in the notebook and the pieces I rely on live somewhere I can test and reuse.

I'm taking it in stages on purpose:

**Phase 1 — Baseline (no documents):** I run the system with nothing loaded into the knowledge base, ask it a small set of test questions drawn from the course materials, and record what comes back. That gives me a clean before-state and shows where the model, left to itself, either doesn't know the answer or makes up something that sounds plausible but isn't backed by anything.

**Phase 2 — Single document:** I add one course PDF to the vector store and ask the same questions again. The point is to see whether and how the answers change, and to start getting a feel for what the retrieval step is actually doing.

**Phase 3 — Expanded knowledge base:** I add the rest of the course documents one at a time and watch how the system behaves as the knowledge base grows — in particular, what retrieval or coherence problems show up once there's more material to choose from.

**Phase 4 — Evaluation:** I score the outputs against a small, structured set of questions with known answers, looking at two things: is the answer actually grounded in the retrieved documents, and does it answer the question I asked? I write up the results honestly, including the ones where the system is still getting it wrong.

I'm only using the course materials as the document knowledge base. I'm not planning to wire in external sources like EIOPA publications or Central Bank of Ireland circulars in this project — that's a longer-term interest, and I'd like to get the basics straight first. Further down the track I'd also like to be able to export my own emails and calendar to Markdown or something similar and try out some of the personal-assistant tools that seem to be trending at the moment, just to see what they're actually like to use.

---

## 2 Solution Development

### 2.1 Data

The knowledge base is just the PDFs the course gives me — lecture materials, readings, and whatever supplementary documents come through the course platform. Nothing from outside that.

I'm doing that on purpose. One thing I took away from the earlier work setting is that an ambitious-looking knowledge base can hide a lot of pipeline problems — if you don't already know the right answer, you can't tell whether retrieval is working. Sticking to material I've worked through myself means I know what the system should return, which makes it much easier to spot when it doesn't. It will also help me learn the course material better.

The prep work breaks into four steps:

1. **Document loading:** I'll load the PDFs using LangChain's document loaders and spot-check that the text came through cleanly — particularly anything with diagrams or formatted tables, where PDF extraction tends to be flakier.
2. **Chunking:** I split each document into smaller text segments before it goes into the vector database. I'll start with the chunk size and overlap the course notebooks suggest and tune them if retrieval looks poor.
3. **Embedding:** I run each chunk through an embedding model to turn it into a numerical vector. I'll use whichever model the course materials demonstrate — keeping the setup close to the reference notebooks makes it easier to reproduce and easier to debug when something goes sideways.
4. **Storage:** The embeddings go into ChromaDB, a local vector database, so the system can fetch the relevant chunks when someone asks a question.

I'm not planning any formal exploratory data analysis beyond eyeballing a sample of chunks to confirm they loaded correctly and read sensibly.

### 2.2 Modelling

The modelling side of this is really two steps wired together with LangChain: retrieval and generation.

Retrieval is the bit that finds the document chunks most relevant to a question. When someone asks something, I run the question through the same embedding model I used on the documents, and the vector store hands back the chunks whose embeddings sit closest to it. I'll start by pulling back a small number per query — probably three to five — and tune from there if the answers look thin or off-topic.

Generation hands the retrieved chunks plus the original question to a language model and asks it to answer using only what's in the context I supplied. I'm running the model locally via Ollama, using one of the smaller open-source models that fits comfortably on a personal laptop. To be clear: I'm picking what runs well on this machine, not making a claim that it's the best model for the job.

I want to be upfront about what this setup can and can't do. A small local model will, in places, give less coherent or less accurate answers than the big commercial ones. Chunking and retrieval settings affect quality in ways I can't always predict from the outside. I'm not trying to build the best possible system here — I'm trying to build one that works well enough that I can evaluate it and learn from where it falls down.

### 2.3 Deployment

I'm running the whole thing on a personal laptop, inside Jupyter notebooks. I'm not planning to host it externally or open it up to anyone else during this project.

That suits the learning goals here. Running locally is easier to debug, raises no data-handling questions, and I can demonstrate it straight off the laptop if anyone wants to see it. Notebooks are also handy for documentation — code, outputs, and the running commentary that explains what I was thinking sit in one place, which means a colleague who isn't a developer can follow what the system does and why I made the choices I did. That ties directly back to the second goal from §1.1: a well-structured notebook is itself a record of the development process that someone non-technical can follow at a high level.

Further down the road — well outside the scope of this capstone — I'd like to adapt this pipeline for internal or regulatory documents in a professional setting. That would mean working through data handling, access controls, and infrastructure questions I haven't touched here.

---

## 3 Evaluation

Evaluation matters to me more than any other part of this project, and it's also the part with the most direct professional carry-over. With RAG systems in practice, the hard question isn't whether the thing runs — it's whether it's actually working well, and if it isn't, being able to say why in terms a stakeholder can follow. That's what I'm trying to set up here.

### 3.1 Evaluation Design

I'm building the evaluation around a small set of test questions taken from the course materials. Because the answers are already sitting in the source documents, I can check two things: did the system retrieve the right material, and is the generated answer consistent with what it retrieved.

I run the same set of questions at each phase — no documents, one document, expanded knowledge base — so I can see directly what adding material to the knowledge base does. The point isn't to present a polished system and claim it works; it's to show the progression from failure to something more useful in a way someone else can follow and check.

Where I can lean on existing tooling, I will. I'm planning to wrap the test questions in a pytest harness with RAGAS-based assertions — that way each evaluation run is just a test suite, results are captured automatically, and a regression in faithfulness or relevancy after a code change shows up the same way a broken unit test would. It also gives me something concrete to point at when I'm explaining to a non-developer how I know the system is or isn't getting better.

### 3.2 Evaluation Results

*To be completed as evaluation runs are executed.*

The plan is to publish the results as a Quarto report rendered directly from the evaluation notebook, with a table per phase showing each test question, the chunks the retriever pulled back, the generated answer, and the RAGAS scores. Failure cases get the same treatment as successes — they're more interesting anyway. Rendering through Quarto means the report regenerates from the underlying notebook and pytest output, so the version in this document and the version on disk can't drift apart. I'd also like to keep the project structured so it can be built with Sphinx as well — not because the project needs two publishing toolchains, but because Sphinx is a skill I want to get more comfortable with, and laying the docstrings, directory layout, and cross-references out in a Sphinx-compatible way from the start costs me very little and pays off the next time I'm dropped into a Python project that uses it.

### 3.3 Evaluation Discussion

*To be completed after evaluation results are in.*

It needs to cover three things:

**Where did the system fail, and why?** The usual suspects are retrieval pulling back the wrong chunks, the language model ignoring or misreading the context I gave it, or the question being phrased in a way the retrieval step couldn't handle. I'll try to put each failure into one of those buckets rather than leaving it as "the answer was bad".

**What would it take to improve things?** Realistically that means tweaking chunking parameters, rewording test questions, or admitting the model just isn't well-suited to certain kinds of query. Some of those are cheap to try; some are signals that the right answer is a different tool.

**What are the limits of the evaluation itself?** RAGAS gives me a number, but it's one measure of quality among several, and a faithfulness score that looks fine on paper can hide problems a human reader would spot in seconds. I'll flag where I think the metrics are telling me less than they appear to.

Finally, I want to come back to the second goal from §1.1 — communicating clearly to people who don't build these systems. The discussion is the place where I prove (or fail to prove) that I can describe what this thing does and doesn't do without either dismissing the failures or overselling the successes.

---

## 4 Legal and Ethical Considerations

The knowledge base is just the course materials, used here for learning. There's no personal data, confidential information, or anything commercially sensitive in it.

I'm running the language model locally via Ollama. Nothing leaves the laptop — no text gets sent to external servers or third-party APIs. That property matters more in a professional setting than it does here, where the documents are public course materials anyway, but it's one of the reasons I picked a local model in the first place: it sidesteps the data-handling questions that come up the moment you start sending documents to a cloud service.

A few broader things are worth flagging even for a prototype like this:

**Accuracy and over-reliance:** a RAG system can give a confident-sounding answer that's just wrong, and the user often can't tell. In a regulated setting, a wrong answer can have real consequences. Any professional version of this would need the limitations spelled out clearly and a human reviewing the outputs — the model isn't the last line of defence.

**Transparency:** users should be able to tell what the system is doing and, just as importantly, what it isn't. Passing off a language model's output as authoritative regulatory guidance would be misleading. That's part of why I'm documenting failure cases as carefully as the successes — if you only show the wins, the reader can't calibrate.

**Knowledge base completeness:** the system's answers are only as good as the documents loaded into it. If something relevant is missing, or chunking strips out the surrounding context, the system can quietly give partial answers — with nothing on screen to flag that anything is missing.

None of these are reasons not to proceed, but they shape how I present the results and the claims I'm willing to make about what the system can do.

---

## 5 Conclusions and Future Work

This document has set out the plan for a small, deliberately scoped RAG prototype built on the course materials. What I want from it is a working understanding of how a RAG pipeline actually behaves, where it falls over, and how to evaluate and talk about its performance in a way that holds up. The secondary goal is to get better at presenting a system like this honestly to a professional audience — including being able to explain a failure without flinching.

I've designed this to be achievable as a part-time learner on a personal laptop. The scope is intentionally narrow: a small document knowledge base, a local language model, and a modest evaluation framework. I'd rather understand a small thing well than gesture at a bigger one — a lesson from the earlier work setting, where building something that looked impressive turned out to be much easier than being able to explain it, defend it, or improve it.

If this goes to plan, the deliverables are a working notebook-based RAG pipeline, a documented evaluation across the development phases, and a reflective write-up of what I learned — about the technology, about its limits, and about how to talk about both.

If I take this further, the directions I'd most like to chase are:

- Pointing the same pipeline at a small set of internal or regulatory documents in a real workplace, to see whether any of this transfers out of a course setting
- Building a lightweight way to record and replay failure cases — the missing piece from the earlier pilot that made it so hard to improve the system over time
- Surfacing some kind of confidence or uncertainty signal to the user, so they have something concrete to calibrate against, rather than just a tone of voice
- Writing a short, plain-English summary of what the system can and can't do — the kind of thing I could hand to a colleague or stakeholder who wasn't in the room while it was being built

---

## References

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *Advances in Neural Information Processing Systems*, 33, 9459–9474.

Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. *arXiv preprint* arXiv:2309.15217.

LangChain Documentation. (2026). Retrieved May 2026 from https://docs.langchain.com

ChromaDB Documentation. (2026). Retrieved May 2026 from https://docs.trychroma.com

Ollama Documentation. (2026). Retrieved May 2026 from https://ollama.com/docs

---

## Appendix II: Lecturer Feedback — Amendments and Clarifications

*Received from lecturer, May 2026. This appendix records the feedback and how it has been incorporated into the project.*

---

### A. Meta-use of AI as a second deliverable

The lecturer noted that the project has two distinct strands worth presenting as separate contributions. The first strand — building and evaluating the RAG pipeline — is already the main subject of this report. The second strand, which the lecturer felt deserves equal recognition, is the **meta-level use of AI**: using Claude Code, GitHub Copilot, and automated agents to manage the project itself — raising GitHub issues, creating milestones, generating commit history, and maintaining structured documentation — in a way that mirrors how a professional might govern a real engineering engagement.

This second strand has direct professional value. Managing a software project via structured issues, branches, and pull requests — whether authored by a human or an AI acting under human direction — is a transferable skill. Demonstrating that an AI assistant can carry out the mechanics of project governance (issue triage, branch strategy, commit conventions), while the human retains the decision-making authority and the audit trail, is a meaningful contribution in its own right, particularly for an actuarial audience who may be asked to oversee AI-assisted work without writing any of the code themselves.

The meta-analysis will be documented separately in a project log (see `log_AI/`) and summarised in the AI Tools appendix of the final submission.

---

### B. Knowledge base: CBI speeches dataset instead of course slides

The course PDFs are slide decks designed to accompany a live lecture. Extracted as plain text, they lose the verbal explanations that give each bullet point its meaning — a RAG system working from them would retrieve fragments that are grammatically complete but contextually thin. The lecturer advised against using them as the primary knowledge base and suggested instead using **the dataset of Central Bank of Ireland (CBI) speeches** that was supplied as part of the course materials.

The CBI speeches are full-text documents: arguments are complete, terminology is used in context, and the corpus covers a domain (financial regulation) that is directly relevant to the professional background motivating this project. This substitution makes the knowledge base substantially richer and makes the evaluation more meaningful — answers can be checked against real, complete statements rather than slide fragments.

This decision supersedes the original plan in §2.1. The data preparation steps (loading, chunking, embedding) are unchanged; only the source documents differ. Where this report refers to "course PDFs", that should be read as "the CBI speeches corpus" from Phase 1 onwards.

---

### C. RAG testing methodology

The lecturer emphasised that evaluation should be approached in a structured, reproducible way rather than run ad hoc. As the project moves into Phases 2–4, each evaluation cycle needs to be set up so that someone reading the outputs can see exactly what was asked, what was retrieved, what was generated, and how the scores were computed — with no black boxes.

A separate Quarto page ([RAG Testing Methodology](../quarto/rag_testing_methodology.qmd)) has been added to the project site to document the planned testing approach before any results are in. This covers: the test set design, the RAGAS metrics used and their limitations, the per-phase progression, and the conventions for recording both successes and failures. Settling this in advance reduces the temptation to adjust methodology in response to results.

---

## Appendix I: Statement on Use of AI Tools

I've used AI tools heavily in this project, and I want to be straightforward about how.

For the report itself, I used Claude (Anthropic) to help structure and draft the written sections from my own notes and rough draft. The underlying ideas, professional experience, and project design are mine; Claude helped me tighten the prose, cut repetition, and land on a tone that sounds like me rather than a textbook. I read every suggestion and decided what stayed, what changed, and what got dropped.

For the code, I plan to lean on GitHub Copilot in much the same way — it's what I use day-to-day at work, and it's genuinely faster than typing everything from scratch. I'll also try Claude Code, partly because I'm curious how it compares and partly because it integrates with Word and Excel in a way I can't get on my work laptop. I'm interested in MCP servers for the same reason — they're blocked at work, so the capstone is a chance to actually use them.

My working principle throughout: the AI tools accelerate the typing and surface options I might not have thought of, but the design choices, the evaluation logic, and the judgments about what's good enough are mine. Where AI-generated code or prose ends up in the final artefact, I'll have read it, understood it, and signed off on it.
