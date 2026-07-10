import os
from qdrant_client import QdrantClient

QDRANT_URL = "YOUR_QDRANT_URL"
QDRANT_API_KEY = "YOUR_QDRANT_API_KEY"
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)

collection_info = client.get_collection(collection_name="lumina_multimodal")
print(f"Total points in collection: {collection_info.points_count}")
