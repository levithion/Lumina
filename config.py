"""Shared configuration for the local memory-search deployment."""

import os
from pathlib import Path


# Set QDRANT_URL (+ optional QDRANT_API_KEY) to use a Qdrant server instead
# of the embedded store; unset means fully local embedded mode.
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "clip-ViT-B-32")
TEXT_MODEL_NAME = os.getenv("TEXT_MODEL_NAME", "all-MiniLM-L6-v2")
# Vision-language model used at ingestion time only; never loaded for search.
CAPTION_MODEL_NAME = os.getenv("CAPTION_MODEL_NAME", "HuggingFaceTB/SmolVLM-500M-Instruct")
CAPTION_ENABLED = os.getenv("CAPTION_ENABLED", "1") not in ("0", "false", "False")

MAX_DOWNLOAD_BYTES = int(os.getenv("MAX_DOWNLOAD_BYTES", str(20 * 1024 * 1024)))
# Perceptual-hash Hamming distance at or below this counts as a near duplicate.
PHASH_NEAR_DUPLICATE_DISTANCE = int(os.getenv("PHASH_NEAR_DUPLICATE_DISTANCE", "8"))
# Score fusion weights per product profile: (visual, semantic_text, exact_text).
# Memory search leans on captions/OCR because screenshots of one app look
# alike visually; the meme profile leans more on raw CLIP similarity.
RETRIEVAL_PROFILE = os.getenv("RETRIEVAL_PROFILE", "memory")
RETRIEVAL_WEIGHTS = {
    "memory": (0.10, 0.50, 0.40),
    "meme": (0.20, 0.45, 0.35),
}

# Local, private memory search: nothing from this mode ever leaves the machine.
# v4 indexes screenshots AND photos; the old screenshot collection name is
# honored when set so existing indexes keep working without migration.
MEMORY_COLLECTION_NAME = os.getenv(
    "MEMORY_COLLECTION_NAME", os.getenv("SCREENSHOT_COLLECTION_NAME", "lumina_memory_v1")
)
SCREENSHOT_COLLECTION_NAME = MEMORY_COLLECTION_NAME  # backward-compat alias
SCREENSHOT_ROOT = os.getenv("SCREENSHOT_ROOT", str(Path.home() / "Screenshots"))
# Embedded Qdrant storage. Used only when QDRANT_URL is not explicitly set;
# the directory doubles as the on-disk database, no Docker required.
MEMORY_STORAGE_PATH = os.getenv(
    "MEMORY_STORAGE_PATH", str(Path(__file__).resolve().parent / "qdrant_data")
)
# Ingestion progress/failures live here so the UI can show library status.
MEMORY_STATE_PATH = os.getenv(
    "MEMORY_STATE_PATH", str(Path(__file__).resolve().parent / ".lumina_state.json")
)
# Local API/UI server.
SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8337"))


def qdrant_client():
    """Prefer an explicitly configured server; otherwise run fully embedded.

    Embedded mode locks the storage directory to one process at a time,
    which is why server.py hosts models, watcher, and DB in a single process.
    ``force_disable_check_same_thread`` lets FastAPI's threadpool share it.
    """
    from qdrant_client import QdrantClient

    if QDRANT_URL:
        if QDRANT_API_KEY:
            return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        return QdrantClient(url=QDRANT_URL)
    return QdrantClient(path=MEMORY_STORAGE_PATH, force_disable_check_same_thread=True)
