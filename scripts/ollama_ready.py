"""
Startup helper: verify Ollama is live and ChromaDB is populated.

Run once after `ollama serve` before opening notebooks/02_query.ipynb:

    python scripts/ollama_ready.py
"""

import sys
import time
import requests
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"
CHROMA_DIR = "./chroma_db"
CSV_PATH = "./inputs/Data/all_ECB_speeches_csv.csv"
COLLECTION = "ecb_speeches"


def wait_for_ollama(timeout: int = 30) -> None:
    print("Checking Ollama... ", end="", flush=True)
    for _ in range(timeout):
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                print(f"running.")
                print(f"  Available models : {models}")
                return
        except Exception:
            time.sleep(1)
    print("FAILED")
    sys.exit("Ollama is not running. Start it with: ollama serve")


def ensure_chromadb() -> None:
    chroma = Path(CHROMA_DIR)
    if chroma.exists() and any(chroma.iterdir()):
        try:
            import chromadb
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            count = client.get_collection(COLLECTION).count()
            print(f"ChromaDB ready.   {count:,} chunks in '{COLLECTION}' collection.")
            return
        except Exception:
            pass  # collection missing — fall through to ingest

    csv = Path(CSV_PATH)
    if not csv.exists():
        sys.exit(
            f"\nCSV not found at {CSV_PATH}.\n"
            "Place the ECB speeches CSV there before running this script."
        )

    print(f"ChromaDB not found. Ingesting corpus — this takes a few minutes...")
    sys.path.insert(0, ".")
    from rag.ingest import ingest_to_chromadb
    ingest_to_chromadb(CSV_PATH, chroma_dir=CHROMA_DIR)
    print("Ingestion complete.")


if __name__ == "__main__":
    wait_for_ollama()
    ensure_chromadb()
    print("\nEnvironment ready. Open notebooks/02_query.ipynb.")
