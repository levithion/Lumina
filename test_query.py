import os
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

QDRANT_URL = "YOUR_QDRANT_URL"
QDRANT_API_KEY = "YOUR_QDRANT_API_KEY"
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)
model = SentenceTransformer('clip-ViT-B-32', device='cpu')

vector = model.encode("a tall building at sunset").tolist()
response = client.query_points(
    collection_name="lumina_multimodal",
    query=vector,
    limit=5
)
for hit in response.points:
    print(hit.payload.get("file_path"), hit.score)
