import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from PIL import Image
from datetime import datetime
import uuid

#1.Setup Device and Model
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

model = SentenceTransformer('clip-ViT-B-32', device = device)

#2. Connect to Qdrant
client = QdrantClient(host = "localhost", port = 6333)
COLLECTION_NAME = "lumina_multimodal"

def ingest_image(image_path):
    print(f"Ingesting: {image_path}...")

    #3. Generate Embedding
    img = Image.open(image_path)
    vector = model.encode(img).tolist()

    #4. Create the 'Point' (The unit of data in a Vector DB)
    point_id = str(uuid.uuid4())
    point = PointStruct(
        id = point_id,
        vector = vector,
        payload = {
            "file_path": image_path,
            "media_type": "image",
            "timestamp": datetime.now().isoformat()
        }
    )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[point]
    )
    print(f"Successfully ingested {image_path} with ID: {point_id}")

if __name__ == "__main__":
    ingest_image("data/test.jpg")

