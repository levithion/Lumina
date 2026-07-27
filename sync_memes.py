"""Fetch fresh memes from configured sources and add them to Qdrant.

Run periodically with cron, a container scheduler, or GitHub Actions:
  python3 sync_memes.py --source imgur
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from config import LIVE_FETCH_LIMIT, LIVE_IMAGE_DIR, MEME_COLLECTION_NAME, qdrant_client
from init_meme_db import initialize_meme_database
from meme_pipeline import MemeEncoder, build_point
from meme_sources import fetch_imgflip, fetch_imgur, fetch_meme_api, fetch_reddit
from storage import enabled as storage_enabled, upload_image


def download(item: dict) -> tuple[str, str] | None:
    url = item["image_url"]
    response = requests.get(url, timeout=30, headers={"User-Agent": "lumina-meme-search/1.0"})
    response.raise_for_status()
    content_type = response.headers.get("content-type", "image/jpeg").split(";")[0]
    extension = {"image/png": ".png", "image/gif": ".gif", "image/webp": ".webp"}.get(content_type, ".jpg")
    digest = hashlib.sha256(response.content).hexdigest()
    path = Path(tempfile.gettempdir()) / f"lumina-{digest}{extension}"
    path.write_bytes(response.content)
    return str(path), f"memes/{digest}{extension}"


def sync_once(source: str, limit: int, client, encoder) -> int:
    if not storage_enabled():
        raise RuntimeError("Set HF_TOKEN and HF_DATASET_REPO for cloud-only ingestion")
    items = []
    if source in ("reddit", "all"):
        try:
            items.extend(fetch_reddit(limit=limit))
        except Exception as exc:
            print(f"Reddit unavailable: {exc}")
    if source in ("imgur", "all"):
        try:
            items.extend(fetch_imgur(limit=limit))
        except Exception as exc:
            print(f"Imgur unavailable: {exc}")
    if source in ("meme_api", "all"):
        try:
            items.extend(fetch_meme_api(limit=limit))
        except Exception as exc:
            print(f"Meme API unavailable: {exc}")
    if source in ("imgflip", "all"):
        try:
            items.extend(fetch_imgflip(limit=limit))
        except Exception as exc:
            print(f"Imgflip unavailable: {exc}")
    directory = Path(LIVE_IMAGE_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    indexed = 0
    manifest_path = directory / "manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as manifest:
        for item in items:
            path = None
            try:
                downloaded = download(item)
                if not downloaded:
                    continue
                path, object_name = downloaded
                image_url = upload_image(path, object_name=object_name)
                point = build_point(path, encoder, template=item.get("template", ""), tags=item.get("tags", []), search_text=item.get("title", ""), metadata={
                    "source": item.get("source"),
                    "source_id": item.get("source_id"),
                    "source_url": item.get("source_url"),
                    "source_image_url": item.get("image_url"),
                    "image_url": image_url,
                    "local_path": "",
                    "title": item.get("title", ""),
                    "subreddit": item.get("subreddit", ""),
                    "source_score": item.get("score", 0),
                    "source_created_at": item.get("created_at"),
                })
                client.upsert(collection_name=MEME_COLLECTION_NAME, points=[point], wait=True)
                manifest.write(json.dumps({"point_id": point.id, **item, "local_path": "", "image_url": image_url}) + "\n")
                indexed += 1
            except Exception as exc:
                print(f"Skipping {item.get('source_url', item.get('image_url'))}: {exc}")
            finally:
                if path:
                    Path(path).unlink(missing_ok=True)
    print(f"Indexed {indexed} fresh memes")
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("reddit", "meme_api", "imgur", "imgflip", "all"), default="meme_api")
    parser.add_argument("--limit", type=int, default=LIVE_FETCH_LIMIT)
    parser.add_argument("--watch", action="store_true", help="repeat continuously")
    parser.add_argument("--interval", type=int, default=900, help="seconds between syncs in watch mode")
    args = parser.parse_args()
    initialize_meme_database()
    client = qdrant_client()
    encoder = MemeEncoder()
    while True:
        try:
            sync_once(args.source, args.limit, client, encoder)
        except Exception as exc:
            print(f"Sync failed: {exc}")
        if not args.watch:
            break
        time.sleep(max(args.interval, 60))


if __name__ == "__main__":
    main()
