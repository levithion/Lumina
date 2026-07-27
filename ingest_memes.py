import argparse
from pathlib import Path

from config import IMAGE_ROOT, MEME_COLLECTION_NAME, qdrant_client
from init_meme_db import initialize_meme_database
from meme_pipeline import ingest_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR and index a meme directory")
    parser.add_argument("folder", nargs="?", default=IMAGE_ROOT)
    parser.add_argument("--template", default="")
    parser.add_argument("--tag", action="append", default=[])
    args = parser.parse_args()
    initialize_meme_database()
    paths = sorted(str(path) for path in Path(args.folder).iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    stats = ingest_paths(qdrant_client(), MEME_COLLECTION_NAME, paths, template=args.template, tags=args.tag)
    print(f"Indexed {stats['processed']} memes; {stats['failed']} failed")


if __name__ == "__main__":
    main()
