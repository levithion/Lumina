# Ingestion pipeline

Lumina's worker fetches memes, deduplicates them twice, captions them with a
vision-language model, and batch-indexes everything in Qdrant.

## Sources

- Reddit Data API for fresh user posts (OAuth)
- `meme-api.com` as a temporary no-key Reddit-backed feed
- Imgur Gallery API for public gallery images (Client-ID required)
- Imgflip API for popular templates

## Deduplication

1. **source_id skip** — before downloading, items already recorded in
   `data/live/manifest.jsonl` are skipped. This makes scheduled runs cheap
   even though meme feeds repeat the same posts.
2. **content-hash IDs** — after download, each image's SHA-256 maps to a
   deterministic Qdrant point ID; existing points are checked in batches and
   duplicates never re-enter embedding or captioning.

## VLM captioning

At ingestion time only, SmolVLM produces a one-sentence caption (including
visible text) plus an `is_sensitive` verdict that powers the UI's safe-content
filter. Captions are embedded into the `semantic_text` vector alongside OCR
text, titles, tags, and template names — this is what lets searches match on
*meaning* rather than exact wording.

Set `CAPTION_ENABLED=0` to ingest without captioning (the disabled captioner
keeps the pipeline running with empty captions).

## Usage

```bash
export HF_TOKEN="hf_..."            # images upload to this public dataset repo
export HF_DATASET_REPO="your-user/lumina-memes"

python3 sync_memes.py --source meme_api --limit 100              # one shot
python3 sync_memes.py --source meme_api --limit 100 --watch      # loop mode
```

Only `manifest.jsonl` remains in `data/live/`; downloaded files are deleted
after indexing.

For Reddit, periodically reconcile deleted posts and remove their Qdrant
points per Reddit's API terms.
