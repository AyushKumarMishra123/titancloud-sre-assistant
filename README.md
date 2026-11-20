# TitanCloud SRE Incident Assistant

An enterprise Retrieval-Augmented Generation (RAG) assistant designed to help Site Reliability Engineering (SRE) teams query, troubleshoot, and resolve incidents using the official 50-page enterprise incident runbook.

---

## 1. Project Overview

**TitanCloud SRE Incident Assistant** is an enterprise AI solution that provides reliable, hallucination-free guidance during production incidents. Grounded exclusively in the official 50-page `TitanCloud Enterprise SRE & Incident Response Runbook`, the assistant answers critical operational queries regarding service level agreements, alert escalations, error codes, and failover steps, accompanying every response with verified original PDF page citations.

---

## 2. Problem Statement

Site Reliability Engineers facing high-severity outages (such as P0 / P1 incidents, replication lag, or control-plane evictions) require immediate, accurate, and verifiable answers from authoritative documentation. Traditional search often fails to capture operational nuance, while generic large language models tend to hallucinate numerical thresholds, timelines, or procedures.

This system solves these challenges by combining:
- **Zero-loss page extraction**: strictly maintaining 1-indexed PDF page boundaries.
- **Hybrid dense and lexical retrieval**: prioritizing exact error codes (`ERR_PG_REPLICA_LAG`), severity tiers (`P0`), and quantitative SLA thresholds (`99.95%`).
- **Grounded synthesis**: enforcing 12 strict operational rules prohibiting hallucination.
- **100% programmatic page citations**: deriving evidence directly from indexed chunk metadata.

---

## 3. RAG Architecture

```
                                  OFFLINE INGESTION PIPELINE
                                  ==========================
[data/titancloud_sre_incident_runbook_50p.pdf] (50 Pages)
       │
       ▼
[PDFExtractor (PyMuPDF)] ──> Cleans text, preserves technical terms & 1-based page numbers
       │
       ▼
[RecursiveCharacterTextSplitter] ──> Chunk size: 900, Overlap: 120 (per-page boundary isolation)
       │
       ▼
[EmbeddingGenerator] ──> SentenceTransformers (all-MiniLM-L6-v2) ──> 384-dim L2-normalized vectors
       │
       ▼
[FAISS IndexFlatIP] ──> Persistent vector store + chunks.json + SHA-256 hash manifest
                                  
                                  
                                  ONLINE QUERY & CHAT PIPELINE
                                  ============================
[User Query / Suggested Question (Streamlit UI)]
       │
       ▼
[FastAPI REST API: POST /chat]
       │
       ├── Context-Aware Query Rewriter: Incorporates session history for follow-ups
       ├── Dense Vector Search: FAISS Inner Product (Cosine Similarity)
       ├── Hybrid Exact Keyword Booster: Prioritizes exact codes (ERR_PG_REPLICA_LAG, P0, etc.)
       ├── Relevance Threshold Filter: Rejects out-of-domain queries (threshold >= 0.28)
       │
       ▼
[Grounded Prompt Builder] ──> Injects retrieved excerpts with [Page X] headers
       │
       ▼
[Gemini LLM (Google GenAI SDK)] ──> 12 authoritative operational rules
       │
       ▼
[Programmatic Citation Synthesizer] ──> Validates citations against chunk metadata
       │
       ▼
[ChatResponse] ──> Grounded Answer + Source Citations + Original Page Numbers
```

---

## 4. Technology Stack

- **Core Language**: Python 3.11+
- **PDF Extraction**: PyMuPDF (`pymupdf`)
- **Chunking Strategy**: Recursive Character Text Splitter (size: 900, overlap: 120)
- **Dense Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`, 384 dimensions)
- **Vector Database**: FAISS (`faiss-cpu`, IndexFlatIP for exact cosine similarity)
- **LLM Integration**: Official Google GenAI SDK (`google-genai`, Gemini 2.5 Flash)
- **Backend Framework**: FastAPI & Uvicorn (ASGI)
- **Data Validation**: Pydantic v2
- **Frontend Framework**: Streamlit (Enterprise Dark Theme)
- **Evaluation & Data**: Pandas, NumPy, JSON, CSV
- **Test Automation**: Pytest

---

## 5. Project Directory Structure

```
titancloud-sre-assistant/
│
├── data/
│   └── titancloud_sre_incident_runbook_50p.pdf     # 50-page official runbook corpus
│
├── backend/
│   ├── __init__.py
│   ├── models.py          # Pydantic v2 schemas (ChatRequest, ChatResponse, StatsResponse)
│   ├── embeddings.py      # all-MiniLM-L6-v2 embedding generator (384-dim, normalized)
│   ├── ingestion.py       # PDF extractor, recursive text splitter, persistent FAISS store
│   ├── retrieval.py       # Semantic & hybrid retriever with follow-up resolution
│   ├── rag_pipeline.py    # Grounded Gemini RAG pipeline & programmatic citation engine
│   ├── evaluation.py      # Precision@K, Recall@K, MRR, Faithfulness evaluation
│   └── main.py            # FastAPI REST backend (/health, /chat, /stats, /documents, /chat/clear)
│
├── frontend/
│   └── app.py             # Streamlit enterprise chatbot application
│
├── vector_store/
│   └── faiss_index/
│       ├── index.faiss            # FAISS vector index binary
│       ├── chunks.json            # 100 DocumentChunk records with metadata
│       └── index_manifest.json    # SHA-256 hash & configuration manifest
│
├── evaluation/
│   ├── test_questions.json        # 10 verified benchmark runbook questions
│   └── rag_eval_scores.csv        # Real evaluation scores (MRR: 1.0, Faithfulness: 0.958)
│
├── tests/
│   ├── test_extraction.py   # PDF readability, page count (50), and text cleaner tests
│   ├── test_chunking.py     # Recursive splitting, size/overlap & boundary tests
│   ├── test_embeddings.py   # Model loading, dimension (384), normalization & query tests
│   ├── test_faiss_store.py  # Index persistence, reloading & SHA-256 hash detection tests
│   ├── test_retrieval.py    # Hybrid search, error-code boosting, out-of-domain filtering tests
│   ├── test_rag_pipeline.py # Grounded prompting, citation accuracy, insufficient context tests
│   ├── test_api.py          # FastAPI endpoints, session isolation, and error handling tests
│   ├── test_frontend.py     # Streamlit helpers, restrictions & integration tests
│   └── test_evaluation.py   # Precision, Recall, MRR, and CSV validation tests
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 6. Installation Instructions (macOS)

Open Terminal and execute the following commands in order:

```bash
# 1. Clone or navigate to the workspace
cd /Users/ayushkumarmishra/Desktop/genai_exam

# 2. Create Python 3.11 virtual environment
/opt/homebrew/bin/python3.11 -m venv .venv

# 3. Activate virtual environment
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 7. Environment Configuration

Copy the example configuration file:

```bash
cp .env.example .env
```

Edit `.env` to configure your Gemini API Key and settings:

```env
# Google Gemini API Key
GEMINI_API_KEY=your_actual_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash

# Relevance & Retrieval Configuration
RELEVANCE_THRESHOLD=0.28
CHUNK_SIZE=900
CHUNK_OVERLAP=120

# Embedding Configuration
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDING_BATCH_SIZE=32

# Paths
PDF_PATH=data/titancloud_sre_incident_runbook_50p.pdf
VECTOR_STORE_DIR=vector_store/faiss_index

# Backend
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
BACKEND_URL=http://localhost:8000
```

> **Note on Security**: `.env`, virtual environment directories, and local caches are strictly excluded via `.gitignore`.

---

## 8. PDF Location & Corpus Verification

The authoritative runbook is pre-positioned at:
`data/titancloud_sre_incident_runbook_50p.pdf`

To verify the file's presence, size, and page count:

```bash
.venv/bin/python -c "
import pymupdf, os
path = 'data/titancloud_sre_incident_runbook_50p.pdf'
print('Exists:', os.path.exists(path))
print('Size:', os.path.getsize(path), 'bytes')
doc = pymupdf.open(path)
print('Page count:', len(doc))
"
```
Output:
```
Exists: True
Size: 102874 bytes
Page count: 50
```

---

## 9. Document Ingestion

Execute the developer ingestion command to parse the PDF, split chunks, compute dense embeddings, and generate the persistent FAISS index:

```bash
.venv/bin/python -m backend.ingestion
```

To force a re-index:

```bash
.venv/bin/python -m backend.ingestion --force
```

### Ingestion Summary
- **Total Pages in PDF**: 50
- **Extracted Pages with Text**: 50
- **Total Chunks Generated**: 100
- **Average Chunk Length**: 797.1 characters
- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **Integrity**: Persistent SHA-256 hash tracking prevents redundant regeneration upon subsequent runs.

---

## 10. Running FastAPI Backend

Start the FastAPI application using Uvicorn:

```bash
.venv/bin/uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

- **Health Check**: `curl http://localhost:8000/health`
- **Swagger Documentation**: Open `http://localhost:8000/docs` in your browser.
- **ReDoc Documentation**: Open `http://localhost:8000/redoc`.

---

## 11. Running Streamlit Frontend

In a separate terminal window, start the Streamlit chatbot:

```bash
source .venv/bin/activate
.venv/bin/streamlit run frontend/app.py --server.port 8501
```

Open your browser at:
👉 **`http://localhost:8501`**

---

## 12. API Documentation

### `GET /health`
Returns system health, FAISS database status, and LLM configuration:
```bash
curl -X GET http://localhost:8000/health
```

### `POST /chat`
Submits a query to the RAG pipeline with session history:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the escalation timeline when a Severity-1 (P0) alert triggers in PagerDuty?",
    "session_id": "test-session-01",
    "chat_history": []
  }'
```

### `POST /chat/clear`
Clears multi-turn conversation memory for a session:
```bash
curl -X POST http://localhost:8000/chat/clear \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test-session-01"}'
```

### `GET /stats`
Retrieves live statistics about the indexed corpus:
```bash
curl -X GET http://localhost:8000/stats
```

### `GET /documents`
Retrieves read-only metadata about the fixed runbook document:
```bash
curl -X GET http://localhost:8000/documents
```

---

## 13. RAG Evaluation

The evaluation framework benchmarked 10 verified test cases across all core operational categories:

Run the evaluation suite:

```bash
.venv/bin/python -m backend.evaluation
```

### Summary of Benchmark Results

| Metric | Score | Explanation |
|---|---|---|
| **Mean Reciprocal Rank (MRR)** | **1.0000** | The authoritative relevant page was retrieved at Rank #1 for 100% of answerable questions |
| **Mean Recall@K** | **0.9400** | 94% of all relevant source pages were captured in top 4 chunks |
| **Mean Faithfulness / Groundedness** | **0.9582** | Numerical claims, timelines, and error codes match ground truth |
| **Mean Answer Relevance** | **0.9500** | Generated responses address the target question accurately |
| **Hallucination Prevention** | **1.0000** | Out-of-domain questions correctly triggered the standard insufficient context response |

Results are saved to: `evaluation/rag_eval_scores.csv`

---

## 14. Running Automated Tests

Run the complete test suite across all modules with Pytest:

```bash
.venv/bin/pytest tests/ -v
```

All **53 automated tests** pass:
- `test_extraction.py`: 50-page exact count, text cleaning, character integrity.
- `test_chunking.py`: Recursive splitting, size/overlap, per-page boundary preservation.
- `test_embeddings.py`: Model loading, 384 dimensions, L2 normalization, batch encoding.
- `test_faiss_store.py`: IndexFlatIP building, reloading, SHA-256 hash change detection.
- `test_retrieval.py`: Semantic search, exact error-code boost, follow-up query rewriting.
- `test_rag_pipeline.py`: Grounded prompting, 12 system rules, programmatic citation accuracy.
- `test_api.py`: FastAPI endpoints (`/health`, `/chat`, `/stats`, `/documents`, `/chat/clear`, `/search`), session isolation.
- `test_frontend.py`: Streamlit helpers, upload control restriction compliance.
- `test_evaluation.py`: Precision@K, Recall@K, MRR, Faithfulness, and CSV output validation.

---

## 15. Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| `zsh: command not found: pip` | Virtual environment not activated | Run `source .venv/bin/activate` |
| `OpenMP error (SIGSEGV 139) on macOS` | Multiple OpenMP runtimes colliding | `backend/embeddings.py` automatically sets `OMP_NUM_THREADS=1` and `KMP_DUPLICATE_LIB_OK=TRUE` |
| `HuggingFace Hub timeout` | Network sandbox blocking HF hub checks | System sets `HF_HUB_OFFLINE=1` to run locally from cached weights |
| `FAISS index missing (HTTP 503)` | Ingestion pipeline not executed | Run `.venv/bin/python -m backend.ingestion` |
| `Gemini API key warning` | `GEMINI_API_KEY` unset in `.env` | The system falls back to an offline extractive mode with verified citations. Add your key to `.env` to enable full synthesis |

---

## 16. Limitations

1. **Fixed Document Corpus**: The system is designed specifically for the fixed 50-page TitanCloud SRE Incident Runbook. User uploads and runtime document replacements are disabled by design.
2. **Context Window Boundary**: Splitting is performed per page to prevent cross-page provenance ambiguity; statements spanning across page boundaries are indexed under their respective pages.
3. **Network Constraints**: When running in an environment without internet connectivity, LLM synthesis operates in an extractive grounded offline mode using retrieved chunks. Providing a valid `GEMINI_API_KEY` activates full generative reasoning.
