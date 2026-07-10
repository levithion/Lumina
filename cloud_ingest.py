import os
import uuid
# pyrefly: ignore [missing-import]
import torch
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

# 1. Cloud Credentials (PASTE YOURS HERE)
QDRANT_URL = "YOUR_QDRANT_URL"
QDRANT_API_KEY = "YOUR_QDRANT_API_KEY"

# 2. Setup Model (Using your M3 GPU)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = SentenceTransformer('clip-ViT-B-32', device=device)

# 3. Connect to Cloud
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)
COLLECTION_NAME = "lumina_multimodal"

def migrate_to_cloud(folder_path, batch_size=20):
    # Create the collection in the cloud if it doesn't exist
    if not client.collection_exists(COLLECTION_NAME):
        print(f"Creating collection {COLLECTION_NAME} in the cloud...", flush=True)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )

    valid_exts = (".jpg", ".jpeg", ".png")
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                   if f.lower().endswith(valid_exts)]
    
    print(f"Found {len(image_paths)} images. Starting cloud upload...", flush=True)

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        points = []

        for path in batch_paths:
            try:
                # Use context manager to properly close image files
                with Image.open(path) as img:
                    # Convert to RGB to prevent issues with RGBA/Grayscale images
                    img = img.convert('RGB')
                    vector = model.encode(img).tolist()
                
                # Create a deterministic UUID based on the file path 
                # so re-running the script doesn't duplicate entries
                import hashlib
                file_hash = hashlib.md5(path.encode('utf-8')).hexdigest()
                deterministic_uuid = str(uuid.UUID(file_hash))
                
                points.append(PointStruct(
                    id=deterministic_uuid,
                    vector=vector,
                    payload={"file_path": path}
                ))
            except Exception as e:
                print(f"Error processing {path}: {e}", flush=True)

        # Upload the batch to Qdrant Cloud
        if points:
            try:
                client.upsert(collection_name=COLLECTION_NAME, points=points)
                print(f"Uploaded batch {i // batch_size + 1}/{(len(image_paths)//batch_size)+1}", flush=True)
            except Exception as e:
                print(f"Error uploading batch {i // batch_size + 1}: {e}", flush=True)

    print("\n✅ Migration Complete! Your images are now in the cloud.", flush=True)

if __name__ == "__main__":
    migrate_to_cloud("data")