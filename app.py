import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from config import CLIP_MODEL_NAME, MEME_COLLECTION_NAME, TEXT_MODEL_NAME, qdrant_client
from meme_pipeline import device_name
from meme_retrieval import hybrid_search

app = FastAPI(title="Lumina Meme Search API", version="2.0")
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
    file_path: str = ""
    ocr_text: str = ""
    template: str = ""
    tags: list[str] = Field(default_factory=list)
    matched_on: list[str] = Field(default_factory=list)


@app.post("/search", response_model=List[SearchResult])
async def search(request: SearchRequest):
    try:
        visual_vector = visual_model.encode(request.query, normalize_embeddings=True).tolist()
        text_vector = text_model.encode(request.query, normalize_embeddings=True).tolist()
        return hybrid_search(
            client, MEME_COLLECTION_NAME, visual_vector, text_vector,
            request.query, request.limit, request.template, request.safe_only,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Meme search failed: {exc}") from exc


@app.get("/health")
def health_check():
    try:
        count = client.count(collection_name=MEME_COLLECTION_NAME, exact=True).count
    except Exception:
        count = 0
    return {"status": "online", "device": device, "indexed_memes": count, "collection": MEME_COLLECTION_NAME}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
