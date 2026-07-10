import os
import uuid
import torch
from datetime import datetime
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

# Hardware setup
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = SentenceTransformer('clip-ViT-B-32', device=device)
client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "lumina_multimodal"

def bulk_ingest(folder_path, batch_size=10):
    # Get all image paths
    valid_exts = (".jpg", ".jpeg", ".png")
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                   if f.lower().endswith(valid_exts)]
    
    print(f"Found {len(image_paths)} images. Using device: {device}")

    points = []
    for path in image_paths:
        try:
            # Generate Embedding
            img = Image.open(path)
            vector = model.encode(img).tolist()
            
            # Prepare Point with metadata
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "file_path": path,
                    "media_type": "image",
                    "timestamp": datetime.now().isoformat()
                }
            ))
            print(f"Processed: {os.path.basename(path)}")
        except Exception as e:
            print(f"Skipping {path}: {e}")

    # Upload in batches to stay under the 32MB payload limit
    if points:
        print(f"\nStarting batch upload (Batch size: {batch_size})...")
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            client.upsert(
                collection_name=COLLECTION_NAME, 
                points=batch
            )
            print(f"Uploaded batch {i // batch_size + 1}")
            
        print(f"\nSuccessfully indexed all {len(points)} images!")

if __name__ == "__main__":
    bulk_ingest("data")