import os
from qdrant_client import QdrantClient

QDRANT_URL = "YOUR_QDRANT_URL"
QDRANT_API_KEY = "YOUR_QDRANT_API_KEY"
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)

records, _ = client.scroll(
    collection_name="lumina_multimodal",
    limit=20,
    with_payload=True,
    with_vectors=False
)
for record in records:
    print(record.payload.get("file_path"))
