import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# 1. Secure Credential Loading (Best Practice)
# This looks for the "Secrets" you saved in Hugging Face Settings
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")

# 2. Setup & Load Model
app = FastAPI(title="Lumina Multimodal Search API")

# On Hugging Face, this will correctly default to "cpu"
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Loading model on {device}...")
model = SentenceTransformer('clip-ViT-B-32', device=device)

# Initialize Client
if QDRANT_API_KEY:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    client = QdrantClient(url=QDRANT_URL)
COLLECTION_NAME = "lumina_multimodal"

# 3. Define Data Models
class SearchRequest(BaseModel):
    query: str
    limit: int = 5

class SearchResult(BaseModel):
    score: float
    file_path: str

@app.post("/search", response_model=List[SearchResult])
async def search(request: SearchRequest):
    try:
        # Encode text query
        vector = model.encode(request.query).tolist()

        # Query Qdrant Cloud
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=request.limit
        )

        return [
            SearchResult(
                score=round(hit.score, 4),
                file_path=hit.payload.get("file_path", "Unknown")
            )
            for hit in response.points
        ]
    except Exception as e:
        print(f"Search Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/health")
def health_check():
    return {"status": "online", "device": device, "indexed_images": 8092}

if __name__ == "__main__":
    import uvicorn
    # Port 8000 is standard for the internal API
    uvicorn.run(app, host='0.0.0.0', port=8000)