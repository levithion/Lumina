"""Fetch fresh memes, deduplicate, caption with a VLM, and batch-index in Qdrant.

Two dedup layers keep scheduled runs cheap even though meme feeds repeat items:
  1. source_id lookup against manifest.jsonl (skips before any download)
  2. content-hash point IDs checked against Qdrant in batches (skips re-embedding)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from captioner import CaptionResult, build_captioner
from config import (
    LIVE_FETCH_LIMIT,
    LIVE_IMAGE_DIR,
    MAX_DOWNLOAD_BYTES,
    MEME_COLLECTION_NAME,
    qdrant_client,
)
from init_meme_db import initialize_meme_database
from meme_pipeline import MemeEncoder, build_point, deterministic_id
from meme_sources import fetch_imgflip, fetch_imgur, fetch_meme_api, fetch_reddit
from storage import enabled as storage_enabled, upload_images

UPSERT_BATCH = 64
ALLOWED_CONTENT_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif", "image/webp": ".webp"}


def download(item: dict[str, Any]) -> tuple[str, str, str]:
    """Download an image with type/size guards; returns (path, repo object name, sha256)."""
    response = requests.get(
        item["image_url"],
        timeout=30,
        stream=True,
        headers={"User-Agent": "lumina-meme-search/1.0"},
    )
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    extension = ALLOWED_CONTENT_TYPES.get(content_type)
    if extension is None:
        raise ValueError(f"unsupported content type: {content_type or 'unknown'}")
    declared_length = response.headers.get("content-length")
    if declared_length and int(declared_length) > MAX_DOWNLOAD_BYTES:
        raise ValueError("file exceeds MAX_DOWNLOAD_BYTES")
    chunks: list[bytes] = []
    received = 0
    for chunk in response.iter_content(chunk_size=1 << 16):
        received += len(chunk)
        if received > MAX_DOWNLOAD_BYTES:
            raise ValueError("file exceeds MAX_DOWNLOAD_BYTES")
        chunks.append(chunk)
    content = b"".join(chunks)
    digest = hashlib.sha256(content).hexdigest()
    path = Path(tempfile.gettempdir()) / f"lumina-{digest}{extension}"
    path.write_bytes(content)
    return str(path), f"memes/{digest}{extension}", digest


def load_known_source_ids(manifest_path: Path) -> set[str]:
    known: set[str] = set()
    if not manifest_path.exists():
        return known
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        source_id = str(entry.get("source_id") or "")
        if source_id:
            known.add(source_id)
    return known


def collect_items(source: str, limit: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    sources = {
        "reddit": fetch_reddit,
        "imgur": fetch_imgur,
        "meme_api": fetch_meme_api,
        "imgflip": fetch_imgflip,
    }
    selected = sources.items() if source == "all" else [(source, sources[source])]
    for name, fetch in selected:
        try:
            items.extend(fetch(limit=limit))
        except Exception as exc:
            print(f"{name} unavailable: {exc}")
    return items


def sync_once(source: str, limit: int, client, encoder: MemeEncoder, captioner) -> dict[str, int]:
    if not storage_enabled():
        raise RuntimeError("Set HF_TOKEN and HF_DATASET_REPO for cloud-only ingestion")
    items = collect_items(source, limit)
    directory = Path(LIVE_IMAGE_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.jsonl"
    known_source_ids = load_known_source_ids(manifest_path)

    stats = {"fetched": len(items), "known": 0, "failed_download": 0, "duplicate": 0, "indexed": 0}
    staged: list[dict[str, Any]] = []
    seen_this_run: set[str] = set()
    for item in items:
        source_id = str(item.get("source_id") or "")
        if source_id and (source_id in known_source_ids or source_id in seen_this_run):
            stats["known"] += 1
            continue
        try:
            path, object_name, digest = download(item)
            seen_this_run.add(source_id)
            staged.append({"item": item, "path": path, "object_name": object_name, "digest": digest})
        except Exception as exc:
            stats["failed_download"] += 1
            print(f"Skipping {item.get('source_url', item.get('image_url'))}: {exc}")

    # Drop anything whose content hash already has a point in Qdrant.
    candidate_ids = [deterministic_id(entry["digest"]) for entry in staged]
    existing_ids: set[str] = set()
    for start in range(0, len(candidate_ids), UPSERT_BATCH):
        records = client.retrieve(
            collection_name=MEME_COLLECTION_NAME,
            ids=candidate_ids[start : start + UPSERT_BATCH],
        )
        existing_ids.update(str(record.id) for record in records)
    fresh = [
        entry
        for entry, point_id in zip(staged, candidate_ids)
        if point_id not in existing_ids
    ]
    stats["duplicate"] = len(staged) - len(fresh)

    indexed_manifest_lines: list[str] = []
    try:
        urls = upload_images([(entry["path"], entry["object_name"]) for entry in fresh])
        for start in range(0, len(fresh), UPSERT_BATCH):
            chunk = fresh[start : start + UPSERT_BATCH]
            points = []
            for entry in chunk:
                item = entry["item"]
                result = CaptionResult()
                try:
                    with Image.open(entry["path"]) as image:
                        if captioner is not None:
                            result = captioner.caption_image(image)
                except Exception as exc:
                    print(f"Captioning failed for {entry['object_name']}: {exc}")
                points.append(
                    build_point(
                        entry["path"],
                        encoder,
                        template=item.get("template", ""),
                        tags=item.get("tags", []),
                        search_text=item.get("title", ""),
                        caption=result.caption,
                        metadata={
                            "source": item.get("source"),
                            "source_id": item.get("source_id"),
                            "source_url": item.get("source_url"),
                            "source_image_url": item.get("image_url"),
                            "image_url": urls.get(entry["object_name"], ""),
                            "local_path": "",
                            "title": item.get("title", ""),
                            "subreddit": item.get("subreddit", ""),
                            "source_score": item.get("score", 0),
                            "source_created_at": item.get("created_at"),
                            "is_sensitive": result.is_sensitive,
                        },
                    )
                )
            if not points:
                continue
            client.upsert(collection_name=MEME_COLLECTION_NAME, points=points, wait=True)
            stats["indexed"] += len(points)
            for entry, point in zip(chunk, points):
                indexed_manifest_lines.append(
                    json.dumps({"point_id": point.id, **entry["item"], "local_path": "", "image_url": urls.get(entry["object_name"], "")}) + "\n"
                )
            with manifest_path.open("a", encoding="utf-8") as manifest:
                manifest.writelines(indexed_manifest_lines)
            indexed_manifest_lines.clear()
    finally:
        for entry in staged:
            Path(entry["path"]).unlink(missing_ok=True)

    print(
        f"Fetched {stats['fetched']} · known {stats['known']} · duplicate {stats['duplicate']} "
        f"· failed {stats['failed_download']} · indexed {stats['indexed']}"
    )
    return stats


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
    captioner = build_captioner()
    while True:
        try:
            sync_once(args.source, args.limit, client, encoder, captioner)
        except Exception as exc:
            print(f"Sync failed: {exc}")
        if not args.watch:
            break
        time.sleep(max(args.interval, 60))


if __name__ == "__main__":
    main()
