# private-rag-assistant

[![Quarto Site](https://img.shields.io/badge/Quarto-Site-blue)](https://almac09.github.io/private-rag-assistant/)

**Live documentation:** https://almac09.github.io/private-rag-assistant/

## Setup

### Prerequisites

- Python ≥3.11
- UV package manager
- Ollama (for local LLM)

### Install Ollama

Install Ollama using winget:

```powershell
winget install Ollama.Ollama -e
```

After installation, pull the required model:

```powershell
ollama pull llama3
```

### Setup Virtual Environment and Dependencies

```powershell
# Create virtual environment
uv venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
uv sync
```

### Run Smoke Test

```powershell
python 00_hello_world.py
```

This checks all LLM providers and confirms setup.