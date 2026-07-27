# Meme search MVP

The meme index is intentionally separate from the existing image collection.

```bash
docker compose up -d
python3 init_meme_db.py
python3 ingest_memes.py data
uvicorn app:app --host 0.0.0.0 --port 8000
streamlit run frontend.py
```

Set `QDRANT_URL`, `QDRANT_API_KEY`, and (for hosted images) `IMAGE_BASE_URL` in
the environment. `IMAGE_BASE_URL` should point at a directory containing the
same filenames as the indexed files; local mode can leave it empty and use
`local_path` instead.

OCR is optional at runtime. If Tesseract is available, caption text is
indexed automatically; without it, visual and semantic template/tag search
still work.
