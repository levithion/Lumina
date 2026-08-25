"""Hybrid retrieval, reverse image search, and reranking for meme search."""

from __future__ import annotations

import re
from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchValue

_TEXT_FIELDS = ("normalized_text", "ocr_text", "caption", "title", "template", "tags", "subreddit")
_VISUAL_WEIGHT = 0.20
_SEMANTIC_WEIGHT = 0.45
_EXACT_WEIGHT = 0.35


def tokenize(value: str) -> list[str]:
    """Lowercase, strip punctuation, and split into word tokens."""
    return re.sub(r"[^\w\s]", " ", value.lower(), flags=re.UNICODE).split()


def _text_score(query: str, payload: dict[str, Any]) -> float:
    """Fraction of query terms present in the indexed text fields.

    Both sides run through the same tokenizer, so punctuation can never split
    a match ("friday" matches "friday.") and substrings cannot fake one
    ("cat" does not match "category").
    """
    query_terms = set(tokenize(query))
    if not query_terms:
        return 0.0
    document = tokenize(" ".join(str(payload.get(key, "")) for key in _TEXT_FIELDS))
    if not document:
        return 0.0
    return len(query_terms.intersection(document)) / len(query_terms)


def build_search_filter(template: str | None = None, safe_only: bool = True) -> Filter | None:
    """Pushdown filters evaluated inside Qdrant, before top-K truncation.

    ``safe_only`` uses ``must_not`` so legacy points that lack ``is_sensitive``
    entirely remain visible instead of being silently excluded.
    """
    must: list[FieldCondition] = []
    must_not: list[FieldCondition] = []
    if template and template.strip():
        must.append(FieldCondition(key="template_key", match=MatchValue(value=template.strip().lower())))
    if safe_only:
        must_not.append(FieldCondition(key="is_sensitive", match=MatchValue(value=True)))
    if not must and not must_not:
        return None
    return Filter(must=must or None, must_not=must_not or None)


def hybrid_search(
    client: Any,
    collection_name: str,
    visual_vector: list[float],
    text_vector: list[float],
    query: str,
    limit: int = 20,
    template: str | None = None,
    safe_only: bool = True,
) -> list[dict[str, Any]]:
    candidate_limit = max(limit * 4, 40)
    query_filter = build_search_filter(template, safe_only)
    common = {
        "collection_name": collection_name,
        "limit": candidate_limit,
        "with_payload": True,
        "query_filter": query_filter,
    }
    sources: dict[str, list[Any]] = {
        "visual": client.query_points(query=visual_vector, using="visual", **common).points,
        "semantic": client.query_points(query=text_vector, using="semantic_text", **common).points,
    }
    weights = {"visual": _VISUAL_WEIGHT, "semantic": _SEMANTIC_WEIGHT}
    merged: dict[Any, dict[str, Any]] = {}
    for source_name, hits in sources.items():
        for hit in hits:
            item = merged.setdefault(
                hit.id,
                {"id": hit.id, "payload": hit.payload or {}, "visual": 0.0, "semantic": 0.0},
            )
            item[source_name] = max(item[source_name], float(hit.score))
    results = []
    for item in merged.values():
        payload = item["payload"]
        exact = _text_score(query, payload)
        score = _VISUAL_WEIGHT * item["visual"] + _SEMANTIC_WEIGHT * item["semantic"] + _EXACT_WEIGHT * exact
        results.append(
            {
                "id": str(item["id"]),
                "score": round(score, 4),
                "image_url": payload.get("image_url", ""),
                "source_url": payload.get("source_url", ""),
                "file_path": payload.get("local_path", ""),
                "ocr_text": payload.get("ocr_text", ""),
                "caption": payload.get("caption", ""),
                "title": payload.get("title", ""),
                "subreddit": payload.get("subreddit", ""),
                "template": payload.get("template", ""),
                "tags": payload.get("tags", []),
                "perceptual_hash": payload.get("perceptual_hash", ""),
                "matched_on": [
                    label
                    for label, value in (
                        ("visual", item["visual"]),
                        ("semantic", item["semantic"]),
                        ("text", exact),
                    )
                    if value > 0
                ],
            }
        )
    return sorted(results, key=lambda result: result["score"], reverse=True)[:limit]


def phash_distance(hash_a: str, hash_b: str) -> int | None:
    """Hamming distance between two hex perceptual hashes; None when incomparable."""
    try:
        return bin(int(str(hash_a), 16) ^ int(str(hash_b), 16)).count("1")
    except (TypeError, ValueError):
        return None


def reverse_image_search(
    client: Any,
    collection_name: str,
    visual_vector: list[float],
    query_phash: str = "",
    limit: int = 20,
    safe_only: bool = True,
    near_duplicate_distance: int = 8,
) -> list[dict[str, Any]]:
    """Find memes visually similar to an uploaded image.

    Perceptual-hash distance flags reposts/near-duplicates; those are promoted
    above plain CLIP-neighbors so the original ranks first.
    """
    query_filter = build_search_filter(safe_only=safe_only)
    hits = client.query_points(
        collection_name=collection_name,
        query=visual_vector,
        using="visual",
        limit=max(limit * 3, 30),
        with_payload=True,
        query_filter=query_filter,
    ).points
    results = []
    for hit in hits:
        payload = hit.payload or {}
        distance = phash_distance(query_phash, payload.get("perceptual_hash", "")) if query_phash else None
        is_exact = distance == 0
        is_near = distance is not None and 0 < distance <= near_duplicate_distance
        results.append(
            {
                "id": str(hit.id),
                "score": round(float(hit.score), 4),
                "phash_distance": distance,
                "duplicate": bool(is_exact or is_near),
                "match_tier": 2 if is_exact else 1 if is_near else 0,
                "image_url": payload.get("image_url", ""),
                "source_url": payload.get("source_url", ""),
                "file_path": payload.get("local_path", ""),
                "ocr_text": payload.get("ocr_text", ""),
                "caption": payload.get("caption", ""),
                "title": payload.get("title", ""),
                "subreddit": payload.get("subreddit", ""),
                "template": payload.get("template", ""),
                "tags": payload.get("tags", []),
            }
        )
    results.sort(key=lambda result: (result["match_tier"], result["score"]), reverse=True)
    return results[:limit]
