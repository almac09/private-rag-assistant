"""
Startup helper: verify Ollama is live, required models are present, and ChromaDB is populated.

Run once after ollama serve before opening notebooks/02_query.ipynb.
Use VS Code F5 (launch.json config) or task 18.
"""

import os
import sys
import time
import requests
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"
COLLECTION = "ecb_speeches"

# Resolve all paths from project root so the script works from any cwd
_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = str(_ROOT / "chroma_db")
CSV_PATH = str(_ROOT / "inputs" / "Data" / "all_ECB_speeches_csv.csv")
sys.path.insert(0, str(_ROOT))

# Load .env to pick up OLLAMA_MODEL
from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

EMBED_MODEL = "nomic-embed-text:latest"
LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
REQUIRED_MODELS = [EMBED_MODEL, LLM_MODEL]


def wait_for_ollama(timeout: int = 30) -> list[str]:
    """Return list of available model names, or exit if Ollama is not running."""
    print("Checking Ollama... ", end="", flush=True)
    for _ in range(timeout):
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                print("running.")
                print(f"  Available models : {models}")
                return models
        except Exception:
            time.sleep(1)
    print("FAILED")
    sys.exit("Ollama is not running. Start it with: ollama serve")


def check_required_models(available: list[str]) -> None:
    """Exit with pull instructions if any required model is missing."""
    missing = [m for m in REQUIRED_MODELS if not any(m in a for a in available)]
    if missing:
        print("\nRequired models not found. Pull them first:\n")
        for m in missing:
            print(f"    ollama pull {m}")
        print()
        sys.exit(1)
    print(f"  Required models OK: {REQUIRED_MODELS}")


def ensure_chromadb() -> None:
    """Ingest the ECB speeches corpus if ChromaDB collection does not exist."""
    chroma = Path(CHROMA_DIR)
    if chroma.exists() and any(chroma.iterdir()):
        try:
            import chromadb
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            count = client.get_collection(COLLECTION).count()
            if count > 0:
                print(f"ChromaDB ready.   {count:,} chunks in '{COLLECTION}' collection.")
                return
            print("ChromaDB collection exists but is empty — re-ingesting...")
        except Exception:
            pass  # collection missing — fall through to ingest

    csv = Path(CSV_PATH)
    if not csv.exists():
        sys.exit(
            f"\nCSV not found at {CSV_PATH}.\n"
            "Place the ECB speeches CSV there before running this script."
        )

    print("ChromaDB not found. Ingesting corpus — this takes a few minutes...")
    from rag.ingest import ingest_to_chromadb
    ingest_to_chromadb(CSV_PATH, chroma_dir=CHROMA_DIR)
    print("Ingestion complete.")


if __name__ == "__main__":
    available = wait_for_ollama()
    check_required_models(available)
    ensure_chromadb()
    print("\nEnvironment ready. Open notebooks/02_query.ipynb.")
