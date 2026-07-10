import os
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

QDRANT_URL = "YOUR_QDRANT_URL"
QDRANT_API_KEY = "YOUR_QDRANT_API_KEY"
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)

# Let's get the top 5 vectors and see if they are identical
response, _ = client.scroll(
    collection_name="lumina_multimodal",
    limit=5,
    with_payload=True,
    with_vectors=True
)

for hit in response:
    print(hit.payload.get("file_path"), "Vector sum:", sum(hit.vector))
