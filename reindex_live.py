"""Rebuild live-source points after changing OCR/search fields."""

import json
from pathlib import Path

from config import LIVE_IMAGE_DIR, MEME_COLLECTION_NAME, qdrant_client
from meme_pipeline import MemeEncoder, build_point
from storage import upload_image


def main() -> None:
    manifest = Path(LIVE_IMAGE_DIR) / "manifest.jsonl"
    if not manifest.exists():
        print(f"Manifest not found: {manifest}")
        return
    client = qdrant_client()
    encoder = MemeEncoder()
    indexed = 0
    seen: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
            path = item.get("local_path", "")
            if not path or path in seen or not Path(path).exists():
                continue
            seen.add(path)
            image_url = upload_image(path)
            point = build_point(
                path,
                encoder,
                template=item.get("template", ""),
                tags=item.get("tags", []),
                search_text=item.get("title", ""),
                metadata={
                    "source": item.get("source"),
                    "source_id": item.get("source_id"),
                    "source_url": item.get("source_url"),
                    "source_image_url": item.get("image_url"),
                    "image_url": image_url,
                    "title": item.get("title", ""),
                    "subreddit": item.get("subreddit", ""),
                    "source_score": item.get("score", 0),
                    "source_created_at": item.get("created_at"),
                },
            )
            client.upsert(collection_name=MEME_COLLECTION_NAME, points=[point], wait=True)
            indexed += 1
        except Exception as exc:
            print(f"Skipping manifest entry: {exc}")
    print(f"Reindexed {indexed} memes")


if __name__ == "__main__":
    main()
