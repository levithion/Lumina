# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Lumina?

Lumina is an AI-powered multimodal image search engine. Users type natural language queries (e.g., "a tall building at sunset") and get back semantically matching images. It uses OpenAI's CLIP model (`clip-ViT-B-32`) via `sentence-transformers` to embed both text and images into a shared 512-dimensional vector space, with Qdrant as the vector database for cosine similarity search.

## Architecture

The system has three layers that run independently:

1. **Qdrant Vector DB** — Runs via Docker (`docker-compose up -d`) on `localhost:6333`. Stores 512-D CLIP embeddings with `file_path` metadata. Collection name: `lumina_multimodal`.

2. **FastAPI Backend** (`app.py`) — Encodes text queries into CLIP vectors and queries Qdrant. Runs on port 8000. Supports both local Qdrant and Qdrant Cloud via `QDRANT_URL` / `QDRANT_API_KEY` env vars.

3. **Streamlit Frontend** (`frontend.py`) — Calls the backend's `POST /search` endpoint, renders results in a 3-column image grid. Runs on port 8501 (default Streamlit).

The frontend never talks to Qdrant directly — it always goes through the FastAPI backend.

## Commands

### Start the stack (local development)

```bash
# 1. Start Qdrant
docker-compose up -d

# 2. First-time only: initialize collection + ingest images from data/
python init_db.py
python bulk_ingest.py

# 3. Run backend and frontend in separate terminals
source venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

source venv/bin/activate
streamlit run frontend.py
```

### CLI search (no frontend needed)

```bash
python search.py
```

### Cloud deployment

`cloud_ingest.py` migrates local image embeddings to Qdrant Cloud. The `Dockerfile` bundles both backend and frontend for Hugging Face Spaces (port 7860).

### Install dependencies

```bash
pip install -r requirement.txt
```

Note: The requirements file is named `requirement.txt` (no 's'), but the Dockerfile references `requirements.txt` — these are out of sync.

## Data Ingestion Pipeline

There are three ingestion scripts, each for a different use case:

| Script | Purpose | Target |
|--------|---------|--------|
| `ingest_data.py` | Single image ingestion | Local Qdrant |
| `bulk_ingest.py` | Batch ingest from `data/` folder (batch_size=10) | Local Qdrant |
| `cloud_ingest.py` | Batch ingest + migrate to Qdrant Cloud (batch_size=20) | Qdrant Cloud |

All ingestion scripts: open image with PIL → encode with CLIP → upsert `PointStruct` with UUID, vector, and `file_path` payload into Qdrant.

Images must be `.jpg`, `.jpeg`, or `.png` and placed in the `data/` directory.

## Key Technical Details

- **Model**: `clip-ViT-B-32` loaded via `SentenceTransformer`, produces 512-D vectors
- **Device selection**: Prefers Apple MPS (`torch.backends.mps.is_available()`), falls back to CPU. No CUDA path exists.
- **Vector distance**: Cosine similarity
- **API endpoints**: `POST /search` (query + limit), `GET /health`
- **Qdrant data persistence**: Mounted at `./qdrant_data` via docker-compose volume

## Known Issues

- `cloud_ingest.py` has hardcoded Qdrant Cloud credentials (URL + API key) that should be moved to environment variables
- `requirement.txt` vs `requirements.txt` naming mismatch between the file and Dockerfile
- The `/health` endpoint has a hardcoded `indexed_images: 8092` instead of querying actual count
- No test suite exists
