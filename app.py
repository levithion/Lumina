import logging
import os
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from config import (
    CLIP_MODEL_NAME,
    MEME_COLLECTION_NAME,
    PHASH_NEAR_DUPLICATE_DISTANCE,
    TEXT_MODEL_NAME,
    qdrant_client,
)
from meme_pipeline import compute_phash, device_name
from meme_retrieval import hybrid_search, reverse_image_search

logger = logging.getLogger("lumina")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Lumina Meme Search API", version="3.0")
device = device_name()
visual_model = SentenceTransformer(CLIP_MODEL_NAME, device=device)
text_model = SentenceTransformer(TEXT_MODEL_NAME, device=device)
client = qdrant_client()


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=20, ge=1, le=100)
    template: Optional[str] = None
    safe_only: bool = True


class SearchResult(BaseModel):
    id: str
    score: float
    image_url: str = ""
    source_url: str = ""
    file_path: str = ""
    ocr_text: str = ""
    caption: str = ""
    title: str = ""
    subreddit: str = ""
    template: str = ""
    tags: list[str] = Field(default_factory=list)
    matched_on: list[str] = Field(default_factory=list)
    perceptual_hash: str = ""
    phash_distance: Optional[int] = None
    duplicate: bool = False


@app.post("/search", response_model=List[SearchResult])
def search(request: SearchRequest):
    try:
        visual_vector = visual_model.encode(request.query, normalize_embeddings=True).tolist()
        text_vector = text_model.encode(request.query, normalize_embeddings=True).tolist()
        return hybrid_search(
            client, MEME_COLLECTION_NAME, visual_vector, text_vector,
            request.query, request.limit, request.template, request.safe_only,
        )
    except Exception:
        logger.exception("meme search failed for query %r", request.query[:100])
        raise HTTPException(status_code=500, detail="Meme search failed") from None


@app.post("/search/image", response_model=List[SearchResult])
def search_image(
    file: UploadFile = File(...),
    limit: int = Query(default=12, ge=1, le=50),
    safe_only: bool = True,
):
    """Reverse meme search: CLIP similarity plus perceptual-hash repost detection."""
    try:
        image = Image.open(file.file).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable image") from None
    try:
        visual_vector = visual_model.encode(image, normalize_embeddings=True).tolist()
        return reverse_image_search(
            client,
            MEME_COLLECTION_NAME,
            visual_vector,
            query_phash=compute_phash(image),
            limit=limit,
            safe_only=safe_only,
            near_duplicate_distance=PHASH_NEAR_DUPLICATE_DISTANCE,
        )
    except Exception:
        logger.exception("image search failed")
        raise HTTPException(status_code=500, detail="Image search failed") from None


@app.get("/health")
def health_check():
    try:
        count = client.count(collection_name=MEME_COLLECTION_NAME, exact=True).count
    except Exception:
        logger.exception("health check could not reach Qdrant")
        return {
            "status": "degraded",
            "device": device,
            "indexed_memes": 0,
            "collection": MEME_COLLECTION_NAME,
            "database": "unreachable",
        }
    return {"status": "online", "device": device, "indexed_memes": count, "collection": MEME_COLLECTION_NAME}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
