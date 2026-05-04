"""
conftest.py
===========
Pytest configuration and fixtures for the RAG assistant tests.
"""

import pytest
import os


@pytest.fixture(scope="session")
def ollama_available():
    """Check if Ollama service is available."""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="session")
def env_loaded():
    """Ensure .env is loaded."""
    from dotenv import load_dotenv
    load_dotenv()
    return True


def pytest_configure(config):
    """Add custom markers for tests."""
    config.addinivalue_line(
        "markers", "requires_ollama: mark test as requiring Ollama service to be running"
    )
    config.addinivalue_line(
        "markers", "requires_api_key(provider): mark test as requiring an API key"
    )
