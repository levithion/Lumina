import os
from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList

QDRANT_URL = os.environ.get("QDRANT_URL", "YOUR_QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "YOUR_QDRANT_API_KEY")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)
COLLECTION_NAME = "lumina_multimodal"

print("Fetching all points...")
offset = None
seen_paths = set()
duplicates_to_delete = []

while True:
    records, next_page_offset = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=1000,
        offset=offset,
        with_payload=True,
        with_vectors=False
    )
    
    for record in records:
        path = record.payload.get("file_path")
        if path in seen_paths:
            duplicates_to_delete.append(record.id)
        else:
            seen_paths.add(path)
            
    if next_page_offset is None:
        break
    offset = next_page_offset

print(f"Found {len(duplicates_to_delete)} duplicate points out of {len(seen_paths) + len(duplicates_to_delete)} total.")

if duplicates_to_delete:
    print("Deleting duplicates...")
    # Delete in batches to avoid payload limits
    batch_size = 500
    for i in range(0, len(duplicates_to_delete), batch_size):
        batch = duplicates_to_delete[i:i+batch_size]
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=PointIdsList(points=batch)
        )
    print("Duplicates successfully deleted!")
else:
    print("No duplicates found.")
