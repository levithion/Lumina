"""Index a local screenshots/photos folder for private semantic search.

Everything stays on your machine: images are never uploaded, embeddings and
VLM captions are computed locally, and points reference absolute file paths.
Content-hash IDs make reruns free — only genuinely new files get processed.
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import SCREENSHOT_COLLECTION_NAME, SCREENSHOT_ROOT, qdrant_client
from init_meme_db import initialize_meme_database
from meme_pipeline import MemeEncoder, build_point, deterministic_id

UPSERT_BATCH = 32
DISCOVER_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}


def discover_images(folder: Path, recursive: bool = True) -> list[str]:
    """List image files under ``folder`` as sorted absolute-path strings."""
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(
        str(path)
        for path in iterator
        if path.is_file() and path.suffix.lower() in DISCOVER_EXTENSIONS
    )


def content_digest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", default=SCREENSHOT_ROOT, help="folder to index (default ~/Screenshots)")
    parser.add_argument("--collection", default=SCREENSHOT_COLLECTION_NAME)
    parser.add_argument("--limit", type=int, default=0, help="cap indexed files (0 = all)")
    parser.add_argument("--no-recursive", action="store_true")
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"Not a folder: {folder}")
    initialize_meme_database(args.collection)
    client = qdrant_client()
    encoder = MemeEncoder()

    try:
        from captioner import build_captioner

        captioner = build_captioner()
    except Exception as exc:
        print(f"Captioning unavailable ({exc}); continuing with OCR + visual search only")
        captioner = None

    paths = discover_images(folder, recursive=not args.no_recursive)
    if args.limit:
        paths = paths[: args.limit]
    print(f"Discovered {len(paths)} images under {folder}")

    stats = {"indexed": 0, "duplicate": 0, "failed": 0}
    pending: list[tuple[str, str]] = []  # (path, digest)
    for index, path in enumerate(paths):
        try:
            pending.append((path, content_digest(path)))
        except OSError as exc:
            stats["failed"] += 1
            print(f"Skipping {path}: {exc}")
        if len(pending) >= UPSERT_BATCH or index == len(paths) - 1:
            ids = [deterministic_id(digest) for _, digest in pending]
            records = client.retrieve(collection_name=args.collection, ids=ids)
            existing = {str(record.id) for record in records}
            fresh = [(p, d) for (p, d), point_id in zip(pending, ids) if point_id not in existing]
            stats["duplicate"] += len(pending) - len(fresh)
            buffer: list[Any] = []
            for path, digest in fresh:
                captured_at = datetime.fromtimestamp(Path(path).stat().st_mtime, tz=timezone.utc).isoformat()
                caption_text = ""
                is_sensitive = False
                try:
                    from PIL import Image as pil_image

                    if captioner is not None:
                        with pil_image.open(path) as source:
                            result = captioner.caption_image(source.convert("RGB"))
                        caption_text, is_sensitive = result.caption, result.is_sensitive
                except Exception as exc:
                    print(f"Captioning failed for {Path(path).name}: {exc}")
                try:
                    buffer.append(
                        build_point(
                            path,
                            encoder,
                            tags=("screenshot",),
                            caption=caption_text,
                            media_type="screenshot",
                            captured_at=captured_at,
                            metadata={"is_sensitive": is_sensitive},
                        )
                    )
                except Exception as exc:
                    stats["failed"] += 1
                    print(f"Skipping {path}: {exc}")
            if buffer:
                client.upsert(collection_name=args.collection, points=buffer, wait=True)
                stats["indexed"] += len(buffer)
                print(f"Indexed {stats['indexed']}/{len(paths) - stats['duplicate'] - stats['failed']}")
            pending.clear()

    print(
        f"Done · indexed {stats['indexed']} · already present {stats['duplicate']} "
        f"· failed {stats['failed']}"
    )


if __name__ == "__main__":
    main()
