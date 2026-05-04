# Create milestones and issues for the RAG capstone project.
# Run this after milestones already exist.
# Usage: .\scripts\github_setup.ps1

$repo = "almac09/private-rag-assistant"

Write-Host "Creating issues..." -ForegroundColor Cyan

$issues = @(
    # Phase 1
    @{ title = "Ingest: Load course PDFs with LangChain PyPDFLoader"; milestone = "Phase 1: Document ingestion"; body = "Implement load_pdfs() in rag/ingest.py using PyPDFLoader" },
    @{ title = "Ingest: Chunk documents with RecursiveCharacterTextSplitter"; milestone = "Phase 1: Document ingestion"; body = "Implement chunk_documents() in rag/ingest.py" },
    @{ title = "Ingest: Embed chunks and store in ChromaDB"; milestone = "Phase 1: Document ingestion"; body = "Implement build_vectorstore() in rag/ingest.py using nomic-embed-text" },
    @{ title = "Ingest: Spot-check sample chunks for quality"; milestone = "Phase 1: Document ingestion"; body = "Load a few PDFs, chunk them, verify text extraction quality" },

    # Phase 2
    @{ title = "Query: Build retrieval chain (top-k similarity search)"; milestone = "Phase 2: RAG query pipeline"; body = "Implement load_vectorstore() and build_rag_chain() in rag/query.py" },
    @{ title = "Query: Wire generation prompt - answer from context only"; milestone = "Phase 2: RAG query pipeline"; body = "Create LLM prompt that constrains answers to retrieved context" },
    @{ title = "Query: Return source chunk citations with answers"; milestone = "Phase 2: RAG query pipeline"; body = "Implement ask() function to return answer + sources" },
    @{ title = "Query: Test baseline vs single-doc retrieval"; milestone = "Phase 2: RAG query pipeline"; body = "Create notebooks/02_query.ipynb - compare answers with no docs vs one PDF" },

    # Phase 3
    @{ title = "Eval: Write gold-standard test_questions.json"; milestone = "Phase 3: Evaluation harness"; body = "Create 20 test questions with known answers from course materials" },
    @{ title = "Eval: Implement RAGAS scoring"; milestone = "Phase 3: Evaluation harness"; body = "Implement run_evaluation() in rag/evaluate.py using RAGAS" },
    @{ title = "Eval: Wrap evaluation in pytest"; milestone = "Phase 3: Evaluation harness"; body = "Create tests/test_evaluation.py with pytest assertions on RAGAS scores" },
    @{ title = "Eval: Publish results in Sphinx/Quarto"; milestone = "Phase 3: Evaluation harness"; body = "Render evaluation_results.json as tables in docs and quarto/" },

    # Phase 4
    @{ title = "Feedback: Log failed queries for replay"; milestone = "Phase 4: Feedback + confidence"; body = "Build lightweight query logging system (question, retrieved chunks, answer)" },
    @{ title = "Feedback: Add confidence/uncertainty signal"; milestone = "Phase 4: Feedback + confidence"; body = "Surface model confidence or RAGAS scores to user" },

    # Docs
    @{ title = "Docs: Verify Sphinx HTML build"; milestone = "Phase 1: Document ingestion"; body = "Run sphinx-build -b html and verify docs/_build/ renders correctly" },
    @{ title = "Docs: Verify Sphinx docx export"; milestone = "Phase 1: Document ingestion"; body = "Run sphinx-build -b docx and verify Word output" },
    @{ title = "Docs: Render Quarto presentation"; milestone = "Phase 1: Document ingestion"; body = "Run quarto render in quarto/ and verify presentation.html" }
)

foreach ($issue in $issues) {
    $params = @(
        "--repo", $repo,
        "--title", $issue.title,
        "--body", $issue.body,
        "--milestone", $issue.milestone,
        "--label", "enhancement"
    )

    gh issue create @params 2>&1 | Out-Null
    Write-Host "  + $($issue.title)"
}

Write-Host "`nDone!" -ForegroundColor Green
Write-Host "View issues: https://github.com/$repo/issues" -ForegroundColor Green
