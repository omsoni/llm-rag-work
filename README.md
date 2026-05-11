# Medical Assistant — RAG-Based Q&A over the Merck Manual

A Retrieval-Augmented Generation (RAG) system that answers clinical questions grounded in the Merck Manual, built with **Meta-Llama-3-8B-Instruct**, **ChromaDB**, and a **cross-encoder reranker**, evaluated by an independent **Mistral-7B** judge LLM, and deployed as a Dockerized Hugging Face Space with a Flask API and Streamlit UI.

## Overview

Healthcare professionals routinely deal with information overload — large manuals, time pressure, and a strong requirement that answers be traceable to a trusted source. This project builds a domain-grounded medical assistant that retrieves the most relevant passages from the Merck Manual (~4,000 pages, 23 sections), reranks them with a cross-encoder for clinical precision, and generates a structured response with Llama-3-8B-Instruct.

The work is split into four notebooks that follow the lifecycle from prompt design → retrieval pipeline → evaluation → deployment.

## The Notebooks

| # | Notebook | What it does |
|---|----------|--------------|
| 1 | [LLM_Medical_Assistant_Prompt_Engineering.ipynb](./LLM_Medical_Assistant_Prompt_Engineering.ipynb) | Prompt engineering as a standalone improvement layer (no RAG). |
| 2 | [RAG_Pipeline_For_Medical_Assistant.ipynb](./RAG_Pipeline_For_Medical_Assistant.ipynb) | The full retrieve → rerank → generate RAG pipeline. |
| 3 | [RAG_Medical_Assistant_Evaluation.ipynb](./RAG_Medical_Assistant_Evaluation.ipynb) | Judge-LLM evaluation across groundedness, relevance, and faithfulness. |
| 4 | [Medical_Assistant_Deployment.ipynb](./Medical_Assistant_Deployment.ipynb) | Flask + Streamlit + Docker deployment to Hugging Face Spaces. |

### 1. Prompt Engineering — [`LLM_Medical_Assistant_Prompt_Engineering.ipynb`](./LLM_Medical_Assistant_Prompt_Engineering.ipynb)

Explores how far prompt engineering alone can push Llama-3-8B-Instruct on medical Q&A, with no RAG and no external knowledge. Five sampling profiles are swept — Deterministic, Conservative, Balanced, Creative, Exploratory — each paired with a tailored prompt template.

Key pieces:

- **Unified prompt builder** using Llama-3's native chat template (`<|begin_of_text|>`, `<|start_header_id|>`, `<|eot_id|>`) with a stable system prompt that includes an explicit "answer ONCE" instruction (the fix for a self-evaluation failure mode that appears when `max_tokens` grows).
- **Heuristic `evaluate_response`** that scores responses on structural quality (bullets, headers), clinical-register vocabulary, safety signals (disclaimer phrases, hedge language, red-flag absolute claims), truncation, and query overlap — rolled up into a `quality_score` in `[0, 1]`. No judge LLM in this notebook.
- **5 configs × 5 fixed queries = 25 runs** aggregated into a config summary, a quality-score matrix, and a best-config-per-query table.

### 2. RAG Pipeline — [`RAG_Pipeline_For_Medical_Assistant.ipynb`](./RAG_Pipeline_For_Medical_Assistant.ipynb)

Builds the full retrieval pipeline over the Merck Manual PDF.

- **PDF ingestion** with PyMuPDF (`fitz`), block-level extraction with whitespace normalization and per-page metadata (page number, source).
- **Chunking** with `RecursiveCharacterTextSplitter` using the `cl100k_base` tokenizer so chunk sizes are token-accurate (default 512 tokens, 50 overlap). No stopword removal or stemming — medical content depends on abbreviations, dosage formats, and hyphenated terms.
- **Embeddings** with `sentence-transformers/all-mpnet-base-v2` (chosen after comparing against `BAAI/bge-large-en-v1.5` and `pritamdeka/S-PubMedBert-MS-MARCO` under T4 VRAM constraints).
- **Vector store**: started with FAISS for in-memory illustration, settled on **ChromaDB** with cosine similarity and persistent storage for production fit.
- **Reranking** with `cross-encoder/ms-marco-MiniLM-L-12-v2` — top-20 retrieved → top-5 reranked.
- **Generation** with `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf` via `llama-cpp-python`, context window 5000 tokens, system prompt that forces context-only answers ("I don't know" otherwise) in a Clinical Explanation / Treatment Protocol format.

### 3. Evaluation — [`RAG_Medical_Assistant_Evaluation.ipynb`](./RAG_Medical_Assistant_Evaluation.ipynb)

Runs five medical queries through multiple chunking and sampling configurations of the RAG pipeline, then scores each response with an independent judge LLM.

- **Judge model**: `Mistral-7B-Instruct-v0.2` (Q4_K_M GGUF) — deliberately different from the generation model to avoid self-bias.
- **Three rubrics** scored per (question, context, answer) triple:
  - **Groundedness** (1–5) — is the answer derived from and supported by the context?
  - **Relevance** (1–5) — does the answer directly and completely address the question?
  - **Faithfulness** (boolean + severity) — is the answer free of unsupported claims?
- Each rubric uses XML-tagged few-shot examples and strict JSON-output instructions.
- **Robust parsing**: `evaluate_with_judge` extracts the first `{...}` block from each rater's output; malformed judge responses are stored as `{"error": "parse_failed", "raw": ...}` rather than crashing the run.
- **Production config selected**: chunk size 512, overlap 64, temperature 0, max_tokens 512.
- Results are aggregated into a per-query comparison table with observations on retrieval quality, chunking, reranking, and prompt-engineering interactions.

### 4. Deployment — [`Medical_Assistant_Deployment.ipynb`](./Medical_Assistant_Deployment.ipynb)

Packages the full stack as a single Docker image on Hugging Face Spaces.

**Architecture:**

```
Streamlit (port 7860) ──HTTP POST──▶ Flask (port 5000)
                                        │
                                        ├─ embed query (mpnet)
                                        ├─ Chroma retrieve top-20
                                        ├─ cross-encoder rerank top-5
                                        └─ llama-cpp-python → Llama-3-8B Q4_K_M
```

- **`api.py`** (Flask) — loads LLM, embedder, reranker, and Chroma collection once at startup. Routes: `GET /health` returns doc count; `POST /query` runs retrieve → rerank → prompt-build → generate. Sampling profiles keyed off temperature (0.0 deterministic through 1.0 exploratory).
- **`streamlit_app.py`** — sidebar slider snaps temperature to the nearest profile, with optional override for `top_p` / `top_k` / `max_tokens`. Includes a debug panel showing payload, status, JSON keys, and raw answer.
- **`Dockerfile`** — `nvidia/cuda:12.1.1-runtime-ubuntu22.04` base, non-root `user` (HF Spaces requirement), prebuilt CUDA wheel for `llama-cpp-python` from `abetlen.github.io`, models pre-downloaded into the image at build time, Chroma DB copied in.
- **`start.sh`** — launches Flask in the background, waits on `/health`, then starts Streamlit on port 7860 with SIGTERM trapped for clean shutdown.

**Live demo:** [`huggingface.co/spaces/omsoni/Medical_Assistant`](https://huggingface.co/spaces/omsoni/Medical_Assistant)

## Tech Stack

| Layer | Choice |
|-------|--------|
| Generation LLM | `meta-llama/Meta-Llama-3-8B-Instruct` (Q4_K_M GGUF via `llama-cpp-python`) |
| Judge LLM | `mistralai/Mistral-7B-Instruct-v0.2` (Q4_K_M GGUF) |
| Embeddings | `sentence-transformers/all-mpnet-base-v2` (768-dim) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-12-v2` |
| Vector store | ChromaDB (persistent, cosine similarity) |
| PDF parsing | PyMuPDF (`fitz`) |
| Chunking | LangChain `RecursiveCharacterTextSplitter` (tiktoken `cl100k_base`) |
| Backend | Flask |
| Frontend | Streamlit |
| Container | Docker (CUDA 12.1.1 / Ubuntu 22.04) |
| Hosting | Hugging Face Spaces |

## Repository Layout

```
.
├── LLM_Medical_Assistant_Prompt_Engineering.ipynb   # Notebook 1 — prompt engineering
├── RAG_Pipeline_For_Medical_Assistant.ipynb         # Notebook 2 — RAG pipeline
├── RAG_Medical_Assistant_Evaluation.ipynb           # Notebook 3 — judge-LLM evaluation
├── Medical_Assistant_Deployment.ipynb               # Notebook 4 — deployment
└── README.md
```

## Reproducing the Project

The notebooks are designed for Google Colab with a T4 GPU. Each notebook installs its own dependencies in the first cells. Recommended order:

1. **Prompt Engineering** — calibrate sampling profiles and the Llama-3 chat-template prompt builder.
2. **RAG Pipeline** — ingest the Merck Manual PDF, build the Chroma collection, and verify retrieve → rerank → generate on the five sample queries.
3. **Evaluation** — load the Mistral judge and score outputs across groundedness, relevance, and faithfulness.
4. **Deployment** — write out `api.py`, `streamlit_app.py`, `Dockerfile`, and `start.sh`, then `HfApi().upload_folder` to a Hugging Face Space.

### Source Document

The knowledge base is the Merck Manuals — a medical reference covering disorders, tests, diagnoses, and drugs, published since 1899. The PDF used in the project is ~4,000 pages across 23 sections.

## Sample Queries

The same five queries are used across all four notebooks to keep comparisons apples-to-apples:

1. *What is the protocol for managing sepsis in a critical care unit?*
2. *What are the common symptoms for appendicitis, and can it be cured via medicine? If not, what surgical procedure should be followed to treat it?*
3. *What are the effective treatments for sudden patchy hair loss (localized bald spots), and what could be the causes?*
4. *What treatments are recommended for a person who has sustained a physical injury to brain tissue?*
5. *What are the necessary precautions and treatment steps for a person who has fractured their leg during a hiking trip?*

## Disclaimer

This project is for educational and research purposes only. The generated responses are not medical advice and must not be used for clinical decision-making. Always consult a qualified healthcare professional.
