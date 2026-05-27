"""
test_setup.py
=============
Tests to verify the environment is correctly set up for the RAG assistant.

Run with: pytest tests/test_setup.py -v
"""

import os
import sys
import subprocess
import pytest
import requests
from dotenv import load_dotenv

load_dotenv()


class TestDependencies:
    """Test that required packages are installed."""

    def test_langchain_installed(self):
        """Check if LangChain is installed."""
        import langchain
        assert langchain.__version__ is not None

    def test_langchain_ollama_installed(self):
        """Check if LangChain Ollama integration is installed."""
        from langchain_ollama import OllamaLLM
        assert OllamaLLM is not None

    def test_chromadb_installed(self):
        """Check if ChromaDB is installed."""
        import chromadb
        assert chromadb.__version__ is not None

    def test_rag_module_importable(self):
        """Check if the RAG module can be imported."""
        import rag
        assert hasattr(rag, "ingest")
        assert hasattr(rag, "query")
        assert hasattr(rag, "evaluate")


def _ollama_running() -> bool:
    try:
        return requests.get("http://localhost:11434/api/tags", timeout=2).status_code == 200
    except Exception:
        return False

ollama_required = pytest.mark.skipif(
    not _ollama_running(), reason="Ollama not running on localhost:11434 — start with: ollama serve"
)


class TestOllamaService:
    """Test that Ollama service is running and accessible."""

    @ollama_required
    def test_ollama_service_running(self):
        """Check if Ollama service is accessible on localhost:11434."""
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        assert response.status_code == 200

    @ollama_required
    def test_ollama_models_available(self):
        """Check if at least one LLM model is available in Ollama."""
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        models = response.json().get("models", [])
        assert len(models) > 0, "No models found in Ollama. Run: ollama pull llama3.2:1b"


class TestOllamaModels:
    """Test that required Ollama models are available."""

    @ollama_required
    def test_default_model_available(self):
        """Check if the default model is available in Ollama."""
        model = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        models = response.json().get("models", [])
        model_names = [m["name"] for m in models]
        model_base = model.split(":")[0]
        available = any(model_base in m for m in model_names)
        assert available, (
            f"Model '{model}' not found. Available models: {model_names}. "
            f"Install with: ollama pull {model}"
        )


class TestLLMProviders:
    """Test that LLM providers can be instantiated."""

    @ollama_required
    def test_ollama_provider(self):
        """Test that Ollama LLM can be instantiated."""
        from langchain_ollama import OllamaLLM
        model = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
        llm = OllamaLLM(model=model)
        assert llm is not None

    def test_openai_provider_optional(self):
        """Test that OpenAI provider can be imported (key is optional)."""
        try:
            from langchain_openai import ChatOpenAI
            assert ChatOpenAI is not None
        except ImportError:
            pytest.skip("OpenAI provider not installed")

    def test_anthropic_provider_optional(self):
        """Test that Anthropic provider can be imported (key is optional)."""
        try:
            from langchain_anthropic import ChatAnthropic
            assert ChatAnthropic is not None
        except ImportError:
            pytest.skip("Anthropic provider not installed")

    def test_gemini_provider_optional(self):
        """Test that Gemini provider can be imported (key is optional)."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            assert ChatGoogleGenerativeAI is not None
        except ImportError:
            pytest.skip("Gemini provider not installed")


class TestEnvironment:
    """Test environment and configuration."""

    def test_python_version(self):
        """Check Python version is >= 3.11."""
        version = sys.version_info
        assert version.major >= 3 and version.minor >= 11, (
            f"Python 3.11+ required, got {version.major}.{version.minor}"
        )

    def test_env_file_exists(self):
        """Check if .env file exists."""
        assert os.path.exists(".env"), ".env file not found in project root"

    def test_data_directory_structure(self):
        """Check basic directory structure."""
        required_dirs = ["rag", "docs", "notebooks", "tests"]
        for dir_name in required_dirs:
            assert os.path.isdir(dir_name), f"Required directory '{dir_name}' not found"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
