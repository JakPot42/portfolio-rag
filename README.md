# Portfolio RAG — Chat With Your Own Work

CLI tool that ingests every README and CLAUDE.md across all 29 portfolio projects and lets Claude answer questions about them — with mandatory source citations. Designed for interview prep: every answer is self-verifiable against source material you already know.

## What it does

- **Indexes** all `README.md` and `CLAUDE.md` files across the portfolio (243 chunks, pure-Python BM25 — no ONNX, no heavy deps)
- **Retrieves** the most relevant sections per query
- **Answers** using only the retrieved context; Claude cites every claim as `[Project — filename]`
- **Refuses** to speculate beyond source text — if the answer isn't in the docs, it says so
- **Bonus:** `transcript` command extracts decisions and action items from meeting notes in CLAUDE.md Decision Points format

## Usage

```bash
# Single-shot query
python main.py query "which project handles fuzzy entity matching"
python main.py query "what was the Pc value in Orbital Sentinel"
python main.py query "summarize every project using a bounded agentic loop"
python main.py query "explain the Claude extracts Z3 decides doctrine"

# Show which chunks were retrieved
python main.py query "GhostTrace deep trace loop" --show-sources

# Control retrieval depth
python main.py query "agentic loop projects" --top-k 8

# Interactive mode
python main.py chat

# Extract decisions from a meeting transcript
python main.py transcript meeting_notes.txt
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then add your ANTHROPIC_API_KEY
python -m pytest       # 35 tests
```

## Architecture

| Layer | Component | Choice |
|---|---|---|
| Retrieval | `bm25_index.py` — pure-Python BM25 (k1=1.5, b=0.75) | Same pattern as P22 Civic RAG — no ONNX, no Torch, works well on technical keyword queries |
| Corpus | `ingest.py` — walks Projects dir, splits on `## ` headers | Each section becomes one chunk with `{project, filename, section, text}` metadata |
| Answer | `claude_chat.py` — Claude Haiku with strict citation system prompt | Citation rule enforced in system prompt: every claim must cite `[Project — file]` |
| CLI | `main.py` — Click + Rich | `query`, `chat`, `transcript` commands |

## Why BM25 over embeddings

The corpus is technical prose where keyword queries dominate: "rapidfuzz", "sgp4", "Z3 UNSAT", "bounded agentic loop". BM25 retrieves these reliably without the 80MB+ ONNX model download that would be needed for vector embeddings. Same judgment call as GhostTrace rejecting ChromaDB's default ONNX embedder.

## Citation discipline

Claude answers with `[Project — filename]` citations for every claim. If the retrieved context doesn't contain the answer, it says "Not found in the indexed portfolio files." — never speculates. This makes the tool trustworthy for interview prep: you can verify every answer against the source docs you already wrote.

## Corpus coverage

- `C:\Users\JakPot\Projects\CLAUDE.md` — master portfolio document (all 29 projects)
- Per-project `README.md` files (16 projects have top-level READMEs)
- Per-project `CLAUDE.md` files (Agora, CFIUS Screener, GhostTrace)
- SEAD 3 `README.md` (lives outside the main Projects dir)

**Tech:** Python, Click, Rich, Anthropic Claude Haiku. No ML framework dependencies.

**CLI only** — no Render deploy. Part of the portfolio as developer tooling, not a web service.
