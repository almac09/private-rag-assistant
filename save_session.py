"""
save_session.py
===============
Saves a Claude conversation session to the /logs folder for your
AI Tools Appendix and personal reference.

Usage:
  1. In Claude.ai, click the conversation menu (⋮) → "Export conversation"
     OR manually copy the full conversation text from the browser
  2. Paste the text into a .txt file OR pipe it directly:

  # Option A — paste from a file:
  python save_session.py --file my_paste.txt --title "Repo setup and hello world"

  # Option B — interactive (prompts you to paste, then press Ctrl+Z Enter on Windows):
  python save_session.py --title "Repo setup and hello world"

Output:
  logs/YYYY-MM-DD_<slug>.md  — formatted markdown ready for appendix
"""

import argparse
import sys
import os
import re
from datetime import date


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:60]


def build_markdown(title: str, raw: str, session_date: str) -> str:
    return f"""# Session Log: {title}

**Date:** {session_date}  
**Tool:** Claude (claude.ai)  
**Project:** Private RAG Assistant — Solvency II Chatbot  
**Author:** Alan McDonagh (A00052777)  

---

## Purpose of This Session

> _(Edit this section to summarise what you were trying to achieve)_

---

## Prompts & Responses

{raw.strip()}

---

## Key Decisions Made This Session

> _(Edit: list the decisions, e.g. "Chose Ollama as primary LLM for data privacy reasons")_

- 
- 
- 

## AI Tool Usage Notes (for Appendix)

- **Tool used:** Claude Sonnet via claude.ai
- **Role:** Project manager, code generation, report writing
- **Prompts provided by:** Alan McDonagh
- **Outputs used for:** _(delete as appropriate)_ Code / Report drafting / Architecture decisions / All three
- **Edits made to AI output:** _(describe any changes you made)_

"""


def main():
    parser = argparse.ArgumentParser(description="Save a Claude session to /logs")
    parser.add_argument("--title", required=True, help="Short title for this session")
    parser.add_argument("--file", default=None, help="Path to a .txt file with the pasted conversation")
    args = parser.parse_args()

    # Read input
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: file not found: {args.file}")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            raw = f.read()
        print(f"Read {len(raw)} characters from {args.file}")
    else:
        print("Paste the conversation below.")
        print("When done, press Ctrl+Z then Enter (Windows) or Ctrl+D (Mac/Linux):\n")
        raw = sys.stdin.read()

    # Build output
    session_date = date.today().isoformat()
    slug = slugify(args.title)
    filename = f"{session_date}_{slug}.md"

    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    output_path = os.path.join(logs_dir, filename)
    content = build_markdown(args.title, raw, session_date)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n✅ Session saved to: logs/{filename}")
    print("Remember to fill in the 'Key Decisions' and 'AI Tool Usage Notes' sections.")
    print("Then: git add logs/ && git commit -m 'Add session log: {args.title}'")


if __name__ == "__main__":
    main()