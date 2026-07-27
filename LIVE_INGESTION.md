# Live meme ingestion

The live worker supports four sources:

- Reddit Data API for fresh user posts (OAuth required)
- `meme-api.com` for a temporary no-key Reddit-backed feed
- Imgur Gallery API for actual public gallery images (Client-ID required)
- Imgflip API for popular templates

Configure environment variables before running:

```bash
export REDDIT_CLIENT_ID="..."
export REDDIT_CLIENT_SECRET="..."
export REDDIT_USER_AGENT="lumina-meme-search/1.0 (by /u/yourname)"
export REDDIT_SUBREDDITS="memes,dankmemes,programmerhumor"
export IMGUR_CLIENT_ID="..."
export IMGUR_GALLERY_PATH="3/gallery/hot/viral/0"
export HF_TOKEN="hf_..."
export HF_DATASET_REPO="your-huggingface-username/lumina-memes"
```

Run a one-time sync:

```bash
python3 sync_memes.py --source meme_api --limit 100
```

After changing search fields, rebuild existing live points with:

```bash
python3 reindex_live.py
```

For an always-on worker, run:

```bash
python3 sync_memes.py --source meme_api --limit 100 --watch --interval 900
```

Alternatively, schedule the one-shot command every 5–15 minutes with cron or a worker scheduler. Keep the
`manifest.jsonl` file: it records source IDs and URLs needed for deduplication,
attribution, and removal handling. For Reddit, periodically reconcile deleted
posts and remove their Qdrant points as required by Reddit’s API terms.

When `HF_TOKEN` and `HF_DATASET_REPO` are set, each image is processed in a
temporary system file, uploaded to the public Hugging Face Dataset repository,
and deleted immediately. Only `manifest.jsonl` remains in `data/live/`; no
persistent meme images are stored in the project folder.
If your Reddit app requires a user-context token, also set `REDDIT_USERNAME`
and `REDDIT_PASSWORD`; otherwise the worker uses an app-only read token.
