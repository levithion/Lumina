"""Meme-specific OCR, metadata, embedding, and indexing helpers."""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

from config import CLIP_MODEL_NAME, IMAGE_BASE_URL, TEXT_MODEL_NAME


def device_name() -> str:
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def extract_ocr_text(image: Image.Image) -> str:
    """Return OCR text when pytesseract and its binary are available.

    Ingestion remains usable without OCR; the point is indexed with an empty
    text field and can still be found through visual similarity.
    """
    try:
        import pytesseract

        text = pytesseract.image_to_string(image)
        return re.sub(r"\s+", " ", text).strip()
    except Exception:
        return ""


def normalize_text(value: str) -> str:
    return re.sub(r"[^\w\s]", " ", value.lower(), flags=re.UNICODE).strip()


def compute_phash(image: Image.Image) -> str:
    """Hex perceptual hash for near-duplicate detection; empty when unavailable."""
    try:
        import imagehash

        return str(imagehash.phash(image))
    except Exception:
        return ""


def image_hashes(image_path: str) -> tuple[str, str]:
    raw = Path(image_path).read_bytes()
    content_hash = hashlib.sha256(raw).hexdigest()
    with Image.open(image_path) as source:
        perceptual_hash = compute_phash(source)
    return content_hash, perceptual_hash


def deterministic_id(content_hash: str) -> str:
    return str(uuid.UUID(content_hash[:32]))


def public_image_url(path: str) -> str:
    if not IMAGE_BASE_URL:
        return ""
    return f"{IMAGE_BASE_URL.rstrip('/')}/{Path(path).name}"


class MemeEncoder:
    def __init__(self) -> None:
        device = device_name()
        self.visual = SentenceTransformer(CLIP_MODEL_NAME, device=device)
        self.text = SentenceTransformer(TEXT_MODEL_NAME, device=device)

    def encode_image(self, image: Image.Image) -> list[float]:
        return self.visual.encode(image, normalize_embeddings=True).tolist()

    def encode_text(self, text: str) -> list[float]:
        return self.text.encode(text or "meme", normalize_embeddings=True).tolist()


def build_point(image_path: str, encoder: MemeEncoder, *, template: str = "", tags: Iterable[str] = (), metadata: dict[str, Any] | None = None, search_text: str = "", caption: str = "", media_type: str = "meme", captured_at: str | None = None) -> PointStruct:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        width, height = image.size
        ocr_text = extract_ocr_text(image)

    content_hash, perceptual_hash = image_hashes(image_path)
    metadata = metadata or {}
    tags_list = sorted({str(tag).strip().lower() for tag in tags if str(tag).strip()})
    template_value = template.strip()
    caption_value = caption.strip()
    search_document = " ".join(
        part for part in (ocr_text, caption_value, template_value, " ".join(tags_list), search_text) if part
    ) or Path(image_path).stem
    payload = {
        "local_path": str(image_path),
        "image_url": public_image_url(image_path),
        "media_type": media_type,
        "ocr_text": ocr_text,
        "normalized_text": normalize_text(ocr_text),
        "caption": caption_value,
        "template": template_value,
        "template_key": template_value.lower(),
        "tags": tags_list,
        # VLM safety verdict; must_not filtering keeps legacy points without
        # this field visible.
        "is_sensitive": bool(metadata.get("is_sensitive", False)),
        "width": width,
        "height": height,
        "content_hash": content_hash,
        "perceptual_hash": perceptual_hash,
        # Screenshots pass their file mtime so results can show real dates;
        # memes keep the ingestion timestamp.
        "created_at": captured_at or datetime.now(timezone.utc).isoformat(),
    }
    payload.update(metadata)
    return PointStruct(
        id=deterministic_id(content_hash),
        vector={
            "visual": encoder.encode_image(image),
            "semantic_text": encoder.encode_text(search_document),
        },
        payload=payload,
    )


def ingest_paths(client: Any, collection_name: str, paths: Iterable[str], *, template: str = "", tags: Iterable[str] = (), batch_size: int = 16) -> dict[str, int]:
    encoder = MemeEncoder()
    points: list[PointStruct] = []
    processed = failed = 0
    for path in paths:
        try:
            points.append(build_point(path, encoder, template=template, tags=tags))
            processed += 1
        except Exception as exc:
            failed += 1
            print(f"Skipping {path}: {exc}")
    for start in range(0, len(points), batch_size):
        client.upsert(collection_name=collection_name, points=points[start : start + batch_size], wait=True)
    return {"processed": processed, "failed": failed}
