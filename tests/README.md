# Tests

This directory contains pytest tests for the private RAG assistant project.

## Running Tests

From VS Code, use the tasks:
- **14. Pytest - Setup Tests** — Run only setup validation tests
- **15. Pytest - All Tests** — Run all tests

Or from the command line:

```bash
# Setup tests only
pytest tests/test_setup.py -v

# All tests
pytest tests/ -v

# Specific test
pytest tests/test_setup.py::TestOllamaService::test_ollama_service_running -v
```

## Test Files

### `test_setup.py`

Validates that the environment is correctly configured:

- **TestDependencies**: Checks required Python packages are installed
  - LangChain, ChromaDB, Ollama integration, RAG module

- **TestOllamaService**: Verifies Ollama is running
  - Service accessible on localhost:11434
  - At least one model available

- **TestOllamaModels**: Confirms required models are available
  - Checks for the default model specified in `.env`

- **TestLLMProviders**: Validates LLM providers can be instantiated
  - Ollama (required)
  - OpenAI, Anthropic, Gemini (optional)

- **TestEnvironment**: Checks environment and configuration
  - Python 3.11+ installed
  - `.env` file exists
  - Required directories exist

## Prerequisites

Before running tests:

1. **Install dependencies**: `uv sync`
2. **Start Ollama**: `ollama serve` (in a separate terminal)
3. **Pull model**: `ollama pull llama3.2:1b` (or your preferred model)

## Troubleshooting

### Tests hang/timeout

If tests hang on Ollama tests, make sure:
- Ollama service is running: `ollama serve`
- Try pinging it: `curl http://localhost:11434/api/tags`

### Model not found

If Ollama tests fail with "Model not found":
```bash
ollama list                    # See what's installed
ollama pull llama3.2:1b        # Install the default model
```

### Dependencies missing

If import tests fail:
```bash
uv sync                        # Reinstall all dependencies
```
