# Run this once after `gh auth login` to create GitHub milestones and issues.
# Usage: .\scripts\github_setup.ps1

$repo = "almac09/private-rag-assistant"

Write-Host "Creating milestones..." -ForegroundColor Cyan

$milestones = @(
    @{ title = "Phase 0: Repo setup"; description = "Hello world per provider, repo scaffolding" },
    @{ title = "Phase 1: Document ingestion"; description = "Load, chunk, embed course PDFs into ChromaDB" },
    @{ title = "Phase 2: RAG query pipeline"; description = "Retrieve + generate + cite sources" },
    @{ title = "Phase 3: Evaluation harness"; description = "RAGAS + pytest gold-standard test set" },
    @{ title = "Phase 4: Feedback + confidence"; description = "Failure logging, uncertainty signal to user" },
    @{ title = "Phase 5: Enterprise deployment plan"; description = "Data handling, access controls, infrastructure notes" }
)

foreach ($m in $milestones) {
    $json = @{
        title = $m.title
        description = $m.description
    } | ConvertTo-Json

    gh api repos/$repo/milestones -H "Accept: application/vnd.github+json" --input - <<< $json 2>&1 | Out-Null
    Write-Host "  + $($m.title)"
}

Start-Sleep -Seconds 2

Write-Host "`nCreating issues..." -ForegroundColor Cyan

# Get milestone number map
$milestoneMap = @{}
$milestonesData = gh api repos/$repo/milestones --paginate -q '.[] | {title, number}'
$milestonesData | ForEach-Object {
    $parts = $_ -split '\s+'
    if ($parts.Count -ge 2) {
        $num = $parts[-1]
        $title = $parts[0..($parts.Count-2)] -join ' '
        $milestoneMap[$title] = [int]$num
    }
}

$issues = @(
    # Phase 1
    @{ title = "Ingest: Load course PDFs with LangChain PyPDFLoader"; milestone = "Phase 1: Document ingestion"; labels = "enhancement" },
    @{ title = "Ingest: Chunk documents with RecursiveCharacterTextSplitter"; milestone = "Phase 1: Document ingestion"; labels = "enhancement" },
    @{ title = "Ingest: Embed chunks with nomic-embed-text and store in ChromaDB"; milestone = "Phase 1: Document ingestion"; labels = "enhancement" },
    @{ title = "Ingest: Spot-check sample chunks for loading quality"; milestone = "Phase 1: Document ingestion"; labels = "enhancement" },

    # Phase 2
    @{ title = "Query: Build retrieval chain (top-k similarity search)"; milestone = "Phase 2: RAG query pipeline"; labels = "enhancement" },
    @{ title = "Query: Wire generation prompt - answer from context only"; milestone = "Phase 2: RAG query pipeline"; labels = "enhancement" },
    @{ title = "Query: Return source chunk citations with each answer"; milestone = "Phase 2: RAG query pipeline"; labels = "enhancement" },
    @{ title = "Query: Notebook 02 - baseline (no docs) vs single-doc comparison"; milestone = "Phase 2: RAG query pipeline"; labels = "enhancement" },

    # Phase 3
    @{ title = "Eval: Write gold-standard test_questions.json (20 questions)"; milestone = "Phase 3: Evaluation harness"; labels = "enhancement" },
    @{ title = "Eval: Implement RAGAS faithfulness + answer relevancy scoring"; milestone = "Phase 3: Evaluation harness"; labels = "enhancement" },
    @{ title = "Eval: Wrap evaluation in pytest assertions"; milestone = "Phase 3: Evaluation harness"; labels = "enhancement" },
    @{ title = "Eval: Publish results table in Sphinx/Quarto report"; milestone = "Phase 3: Evaluation harness"; labels = "enhancement" },

    # Phase 4
    @{ title = "Feedback: Log failed queries with question + retrieved chunks + answer"; milestone = "Phase 4: Feedback + confidence"; labels = "enhancement" },
    @{ title = "Feedback: Surface confidence/uncertainty signal to user"; milestone = "Phase 4: Feedback + confidence"; labels = "enhancement" },

    # Docs
    @{ title = "Docs: Verify Sphinx HTML build works end-to-end"; milestone = "Phase 1: Document ingestion"; labels = "documentation" },
    @{ title = "Docs: Verify Sphinx docx export works end-to-end"; milestone = "Phase 1: Document ingestion"; labels = "documentation" },
    @{ title = "Docs: Render Quarto presentation from quarto/ directory"; milestone = "Phase 1: Document ingestion"; labels = "documentation" }
)

foreach ($issue in $issues) {
    $milestoneNum = $milestoneMap[$issue.milestone]
    if ($milestoneNum) {
        $json = @{
            title = $issue.title
            milestone = $milestoneNum
            labels = @($issue.labels)
        } | ConvertTo-Json

        gh api repos/$repo/issues -H "Accept: application/vnd.github+json" --input - <<< $json 2>&1 | Out-Null
        Write-Host "  + $($issue.title)"
    }
}

Write-Host "`nDone! Visit: https://github.com/$repo/issues" -ForegroundColor Green
Write-Host "Create a Project board at: https://github.com/$repo/projects" -ForegroundColor Green
