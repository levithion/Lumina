"""Hybrid retrieval and reranking for meme search."""

from __future__ import annotations

import re
from typing import Any


def _text_score(query: str, payload: dict[str, Any]) -> float:
    q = re.sub(r"[^\w\s]", " ", query.lower()).strip()
    if not q:
        return 0.0
    text = " ".join(
        str(payload.get(key, "")) for key in ("normalized_text", "ocr_text", "title", "template", "tags", "subreddit")
    ).lower()
    if q in text:
        return 1.0
    terms = set(q.split())
    available = set(text.split())
    return len(terms & available) / max(len(terms), 1)


def hybrid_search(client: Any, collection_name: str, visual_vector: list[float], text_vector: list[float], query: str, limit: int = 20, template: str | None = None, safe_only: bool = True) -> list[dict[str, Any]]:
    candidate_limit = max(limit * 4, 40)
    visual = client.query_points(collection_name=collection_name, query=visual_vector, using="visual", limit=candidate_limit, with_payload=True).points
    semantic = client.query_points(collection_name=collection_name, query=text_vector, using="semantic_text", limit=candidate_limit, with_payload=True).points
    merged: dict[Any, dict[str, Any]] = {}
    for source, weight in ((visual, 0.20), (semantic, 0.45)):
        for hit in source:
            payload = hit.payload or {}
            if template and str(payload.get("template", "")).lower() != template.lower():
                continue
            if safe_only and payload.get("is_sensitive", False):
                continue
            item = merged.setdefault(hit.id, {"id": hit.id, "payload": payload, "visual": 0.0, "semantic": 0.0})
            key = "visual" if weight < 0.3 else "semantic"
            item[key] = max(item[key], float(hit.score))
    results = []
    for item in merged.values():
        exact = _text_score(query, item["payload"])
        score = 0.20 * item["visual"] + 0.45 * item["semantic"] + 0.35 * exact
        payload = item["payload"]
        results.append({
            "id": item["id"], "score": round(score, 4),
            "image_url": payload.get("image_url", ""),
            "file_path": payload.get("local_path", ""),
            "ocr_text": payload.get("ocr_text", ""),
            "template": payload.get("template", ""),
            "tags": payload.get("tags", []),
            "matched_on": [key for key, value in (("visual", item["visual"]), ("semantic_text", item["semantic"]), ("ocr", exact)) if value > 0],
        })
    return sorted(results, key=lambda result: result["score"], reverse=True)[:limit]
