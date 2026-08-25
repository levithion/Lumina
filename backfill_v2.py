"""Migrate legacy points into the VLM-captioned collection.

For every point in the legacy collection this script fetches the hosted image,
generates a SmolVLM caption plus safety verdict, re-embeds the enriched search
document, and upserts into the target collection. Content-hash IDs are stable,
so reruns skip anything already migrated (free resume).
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator

import requests
from PIL import Image

from captioner import CaptionResult, build_captioner
from config import LEGACY_COLLECTION_NAME, MAX_DOWNLOAD_BYTES, MEME_COLLECTION_NAME, qdrant_client
from init_meme_db import initialize_meme_database
from meme_pipeline import MemeEncoder, build_point

BATCH_SIZE = 32
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def iter_points(client: Any, collection_name: str, batch_size: int = BATCH_SIZE) -> Iterator[Any]:
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
        )
        yield from points
        if offset is None or not points:
            return


def fetch_image(url: str) -> bytes:
    response = requests.get(url, timeout=30, headers={"User-Agent": "lumina-meme-search/1.0"})
    response.raise_for_status()
    if len(response.content) > MAX_DOWNLOAD_BYTES:
        raise ValueError("file exceeds MAX_DOWNLOAD_BYTES")
    return response.content


def extension_for(payload: dict[str, Any]) -> str:
    url = str(payload.get("image_url") or payload.get("source_image_url") or "")
    suffix = Path(url.split("?")[0]).suffix.lower()
    return suffix if suffix in IMAGE_EXTENSIONS else ".jpg"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-collection", default=LEGACY_COLLECTION_NAME)
    parser.add_argument("--target-collection", default="")
    parser.add_argument("--limit", type=int, default=0, help="cap migrated points (0 = all)")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-upsert points even when they already exist in the target (enrichment mode)",
    )
    args = parser.parse_args()

    target_collection = args.target_collection or MEME_COLLECTION_NAME
    initialize_meme_database(target_collection)
    client = qdrant_client()
    encoder = MemeEncoder()
    captioner = build_captioner()

    migrated = skipped_existing = failed = total = 0
    buffer: list[Any] = []

    def flush() -> None:
        nonlocal migrated, skipped_existing
        if not buffer:
            return
        pending = buffer
        if not args.overwrite:
            records = client.retrieve(
                collection_name=target_collection,
                ids=[point.id for point in buffer],
            )
            existing = {str(record.id) for record in records}
            pending = [point for point in buffer if str(point.id) not in existing]
            skipped_existing += len(buffer) - len(pending)
        if pending:
            client.upsert(collection_name=target_collection, points=pending, wait=True)
            migrated += len(pending)
        buffer.clear()

    for record in iter_points(client, args.source_collection):
        if args.limit and total >= args.limit:
            break
        payload = record.payload or {}
        image_url = payload.get("image_url") or payload.get("source_image_url") or ""
        total += 1
        if not image_url:
            failed += 1
            print(f"Skipping {record.id}: no hosted image URL")
            continue
        try:
            content = fetch_image(str(image_url))
        except Exception as exc:
            failed += 1
            print(f"Skipping {record.id}: download failed ({exc})")
            continue
        result = CaptionResult()
        temp_path: Path | None = None
        try:
            descriptor, temp_name = tempfile.mkstemp(suffix=extension_for(payload))
            os.close(descriptor)
            temp_path = Path(temp_name)
            temp_path.write_bytes(content)
            with Image.open(temp_path) as source:
                rgb = source.convert("RGB")
                if captioner is not None:
                    result = captioner.caption_image(rgb)
                point = build_point(
                    temp_path,
                    encoder,
                    template=str(payload.get("template", "")),
                    tags=payload.get("tags", []) or [],
                    search_text=str(payload.get("title", "")),
                    caption=result.caption,
                    metadata={
                        "source": payload.get("source"),
                        "source_id": payload.get("source_id"),
                        "source_url": payload.get("source_url"),
                        "source_image_url": payload.get("source_image_url"),
                        "image_url": payload.get("image_url"),
                        "local_path": "",
                        "title": payload.get("title", ""),
                        "subreddit": payload.get("subreddit", ""),
                        "source_score": payload.get("source_score", 0),
                        "source_created_at": payload.get("source_created_at"),
                        "is_sensitive": result.is_sensitive,
                    },
                )
            buffer.append(point)
        except Exception as exc:
            failed += 1
            print(f"Skipping {record.id}: processing failed ({exc})")
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)
        if len(buffer) >= BATCH_SIZE:
            flush()

    flush()
    print(f"Migrated {migrated} · already present {skipped_existing} · failed {failed} · scanned {total}")


if __name__ == "__main__":
    main()
