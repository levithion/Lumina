# Deployment

Two supported modes:

## Cloud (recommended)

Free-tier services only:

1. **Streamlit Community Cloud** runs `streamlit_app.py` (UI + inference).
2. **Qdrant Cloud** stores the named visual/semantic vectors.
3. **Hugging Face Datasets** hosts meme image files publicly.
4. **GitHub Actions** is the scheduled ingestion worker.

Repository secrets (**Settings → Secrets and variables → Actions**):

```text
QDRANT_URL
QDRANT_API_KEY
HF_TOKEN
HF_DATASET_REPO=Shshank/lumina-memes
```

The sync workflow runs every 15 minutes (manual runs available under
**Actions → Sync fresh memes → Run workflow**) with a concurrency guard and a
persistent Hugging Face model cache so repeated runs stay fast.

Streamlit secrets for Community Cloud: add `QDRANT_URL` and `QDRANT_API_KEY`.

## Local decoupled mode

```bash
docker compose up -d          # local Qdrant on :6333
python3 init_meme_db.py       # create collection + payload indexes
uvicorn app:app --port 8000   # FastAPI backend (loads CLIP + MiniLM)
streamlit run frontend.py     # UI on :8501
```

The backend exposes `POST /search` (text) and `POST /search/image`
(multipart reverse search). `GET /health` reports `"degraded"` when Qdrant is
unreachable instead of pretending to be online.

## Migrating from the legacy index (v1)

The active collection default is `lumina_memes_v2`. Populate it from your
existing v1 points without redownloading anything:

```bash
python3 backfill_v2.py                 # full migration, resumable by design
python3 backfill_v2.py --limit 3000    # or cap to ship the demo faster
```

Rollback at any time: set `MEME_COLLECTION_NAME=lumina_memes_v1`.
