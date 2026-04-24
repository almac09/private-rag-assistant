"""
00_hello_world.py
=================
Tests each LLM provider in order of preference:
  1. Ollama      (local — no key needed, preferred for privacy)
  2. OpenAI      (API key optional)
  3. Anthropic   (API key optional)
  4. Gemini      (API key optional)

A provider is skipped gracefully if:
  - No API key is found in .env
  - The service is not reachable (e.g. Ollama not running)

Run with:
  python 00_hello_world.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

PROMPT = "In one sentence, what is Solvency II?"

results = {}


# ── 1. OLLAMA (local, always tried first) ──────────────────────────────────
def test_ollama():
    model = os.getenv("OLLAMA_MODEL", "llama3")
    try:
        from langchain_ollama import OllamaLLM
        llm = OllamaLLM(model=model)
        response = llm.invoke(PROMPT)
        return response.strip()
    except Exception as e:
        return f"SKIPPED — {e}"


# ── 2. OPENAI ──────────────────────────────────────────────────────────────
def test_openai():
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return "SKIPPED — no OPENAI_API_KEY in .env"
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=key)
        response = llm.invoke(PROMPT)
        return response.content.strip()
    except Exception as e:
        return f"SKIPPED — {e}"


# ── 3. ANTHROPIC ───────────────────────────────────────────────────────────
def test_anthropic():
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return "SKIPPED — no ANTHROPIC_API_KEY in .env"
    try:
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model="claude-sonnet-4-6", api_key=key)
        response = llm.invoke(PROMPT)
        return response.content.strip()
    except Exception as e:
        return f"SKIPPED — {e}"


# ── 4. GEMINI ──────────────────────────────────────────────────────────────
def test_gemini():
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not key:
        return "SKIPPED — no GOOGLE_API_KEY in .env"
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=key)
        response = llm.invoke(PROMPT)
        return response.content.strip()
    except Exception as e:
        return f"SKIPPED — {e}"


# ── RUN ALL TESTS ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Private RAG Assistant — Provider Hello World")
    print(f"Prompt: '{PROMPT}'")
    print("=" * 60)

    providers = [
        ("Ollama (local)", test_ollama),
        ("OpenAI",         test_openai),
        ("Anthropic",      test_anthropic),
        ("Gemini",         test_gemini),
    ]

    for name, fn in providers:
        print(f"\n[{name}]")
        result = fn()
        print(result)

    print("\n" + "=" * 60)
    print("Done. Any provider showing a real answer is ready to use.")
    print("Install Ollama from https://ollama.com if you want local mode.")
    print("=" * 60)
