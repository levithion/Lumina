from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(host = "localhost", port = 6333)

COLLECTION_NAME = "lumina_multimodal"

def initialize_database():
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"Creating collection: {COLLECTION_NAME}...")

        client.create_collection(collection_name = COLLECTION_NAME, vectors_config=VectorParams(size = 512, distance = Distance.COSINE),)
        print("Collection Created Successfully!")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists. Ready to go!")

if __name__ == "__main__":
    initialize_database()