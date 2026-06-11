# Simple RAG

**A fully local, auditable RAG pipeline — no frameworks, no cloud, no PyTorch.**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Runs Locally](https://img.shields.io/badge/inference-100%25%20local-orange)
![No Frameworks](https://img.shields.io/badge/LangChain-not%20here-lightgrey)

Ask questions about your PDFs and documents through a clean chat UI and get answers with **page-level citations you can actually verify**. Documents, embeddings, retrieval, and generation never leave your machine.

The entire pipeline lives in **one readable Python file**, built from raw SDKs — no LangChain, no LlamaIndex, no Haystack. You can audit every prompt, every score, every fusion step in an afternoon. That's the point.

<!-- Add a screenshot or demo GIF here: ![Demo](docs/demo.png) -->

## Why this exists

Most RAG stacks hide the interesting parts behind framework abstractions, and most "local" tools still phone home for embeddings or generation. Simple RAG does neither:

- **Auditable** — one file, five labeled blocks, explicit math. Every answer archives its raw retrieved fragments, scores, model, and timing to JSON.
- **Actually local** — FastEmbed (ONNX) embeddings, in-file BM25, ONNX cross-encoder reranking, Ollama generation. Runs on a CPU-only laptop.
- **Honest** — real relevance scores from a cross-encoder (not vector geometry), per-source confidence, and a generator instructed to refuse when context is irrelevant.

## Highlights

- **Hybrid retrieval** — dense vector search (FastEmbed ONNX) + BM25 lexical search (implemented in-file, zero dependencies), fused with reciprocal rank fusion.
- **Cross-encoder reranking** — an 80 MB ONNX cross-encoder rescores every candidate by reading the query and chunk together. Confidence scores are real relevance, not vector geometry.
- **Contextual chunk headers** — each document gets an LLM-generated summary prepended to every chunk's embedding, so context-dependent chunks stay retrievable.
- **Document overview chunks** — document-level questions ("what is this paper about?") get a real retrieval target, with automatic fallback when nothing else is relevant.
- **Page-cited answers** — `[Source Chunk 2, p. 38]`, traced from PDF page markers through chunking to generation.
- **OCR-aware loading** — PyMuPDF native text + table extraction + Tesseract OCR for scanned pages. Handles PDF, DOCX, TXT, MD, and images.
- **Ingest once, ask many** — projects keep a persistent index; unchanged files are never re-parsed (byte-hash manifest).
- **Claude-style web UI** — projects, document upload, chat sessions, expandable source views, live timers. Vanilla JS, no build step.

## Architecture

```
                          ┌─────────────────────────────┐
 PDF / DOCX / TXT / IMG ─▶│ Loader (PyMuPDF + Tesseract)│
                          └──────────────┬──────────────┘
                                         │ text + [Page N] markers
                          ┌──────────────▼──────────────┐
                          │ Recursive semantic chunker  │
                          │ + page mapping              │
                          │ + contextual headers (LLM)  │
                          │ + document overview chunk   │
                          └──────────────┬──────────────┘
                                         │ chunks (content-addressed IDs)
                          ┌──────────────▼──────────────┐
                          │ FastEmbed (ONNX) → ChromaDB │
                          └──────────────┬──────────────┘
                                         │
        question ──┬──── dense top-20 ───┤
                   └──── BM25 top-20 ────┤
                                         ▼
                          ┌─────────────────────────────┐
                          │ Reciprocal rank fusion      │
                          │ → ONNX cross-encoder rerank │
                          │ → threshold + overview      │
                          │   fallback                  │
                          └──────────────┬──────────────┘
                                         │ top-k chunks + confidences
                          ┌──────────────▼──────────────┐
                          │ Ollama (local LLM)          │
                          │ grounded prompt, page-cited │
                          └──────────────┬──────────────┘
                                         ▼
                          answer + citations + audit JSON
```

Everything lives in [`local_rag_pipeline.py`](local_rag_pipeline.py) (pipeline) and [`webapp/`](webapp/) (FastAPI + vanilla JS UI).

## Quick start

```bash
# Python dependencies
pip install -r requirements.txt

# System dependencies
brew install tesseract        # OCR engine
ollama pull qwen2.5:7b        # local generation model
```

### Web UI

```bash
python3 webapp/server.py
```

Opens http://127.0.0.1:8400 automatically. Create a project, add documents, ask questions. Uploads are instant — indexing (parse → OCR → chunk → summarize → embed) runs when you ask your first question, with live status.

### CLI

```bash
# Build a project index from a file or folder
python3 local_rag_pipeline.py -p myproject -i ./docs/paper.pdf

# Ask as many questions as you want (no re-chunking)
python3 local_rag_pipeline.py -p myproject -a "what methodology was used?"

# Rebuild from scratch (e.g. after changing chunking)
python3 local_rag_pipeline.py -p myproject -i ./docs --reindex

# Self-contained demo, no files needed
python3 local_rag_pipeline.py --demo
```

| Flag | Meaning |
|------|---------|
| `-p` | project name |
| `-i` | ingest file or folder |
| `-a` | ask a question |
| `-s` | session name |
| `-n` | number of results |
| `-m` | Ollama model |
| `--chunk-size` / `--chunk-overlap` | chunking parameters |
| `--reindex` | rebuild index from scratch |
| `--demo` | self-contained demo |

## Benchmarks

Measured on a **2020 Intel MacBook Air (CPU-only, 16 GB)** — roughly the worst hardware you'd realistically run this on:

| Metric | Result |
|--------|--------|
| Answer time after chunker fix + prompt-fitted context | **10m47s → 5m46s** (same question), confidence 0.85 → 0.90 |
| Hybrid + reranking separation | relevant chunks **0.99+**, irrelevant **< 0.62** — clean split L2 distance never gave |
| Reranker overhead | ~0.3 s warm load, < 1 s to score 40 candidates |
| 96-page thesis ingest | parse/OCR 19 s, 241 chunks, embed+index ~2 min — once |

## Design decisions

**Why no frameworks?** Every stage is explicit and inspectable: you can read the exact prompt, the exact fusion math, the exact score normalization. The whole pipeline is one file you can audit in an afternoon.

**Why ONNX everywhere?** FastEmbed runs BGE embeddings and the cross-encoder reranker on ONNX Runtime — no PyTorch, small downloads, fast cold starts, runs on a CPU-only laptop.

**Why a reranker on a slow machine?** It's nearly free (< 1 s per question) next to local 7B generation, and it's the single biggest quality lever: hybrid recall casts a wide net, the cross-encoder makes the final call. Threshold is deliberately moderate (0.25) — strict filtering discards context the generator needs.

**Why contextual headers and overview chunks?** Bare chunks lose their document context ("Pros and Cons" of *what*?). Prepending a document summary to each chunk's embedding — and indexing the summary itself as a retrievable overview — makes both pinpoint and document-level questions answerable. (Both techniques validated against a 2025 master's thesis on enterprise RAG that this repo was tested with.)

**Honest confidence and refusals.** The generator is instructed to refuse when context is genuinely irrelevant, confidences are shown per-source, and every raw fragment is archived to JSON so you can audit any answer.

## Project layout

```
local_rag_pipeline.py    # the entire pipeline, 5 labeled blocks
webapp/
  server.py              # FastAPI JSON API wrapping the pipeline
  static/                # Claude-style chat UI (vanilla JS)
projects/<name>/         # per-project index, manifest, chat history (gitignored)
docs/                    # sample input documents
DEVLOG.md                # full development history, every change explained
```

## Development log

The complete build history — every bug, fix, measurement, and design change — is in [DEVLOG.md](DEVLOG.md). It reads as a narrative of how the system evolved from basic vector search to a hybrid, reranked, page-cited pipeline.

## License

MIT — see [LICENSE](LICENSE).
