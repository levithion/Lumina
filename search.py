import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# 1. Setup Device & Model (Matches ingestion for consistency)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = SentenceTransformer('clip-ViT-B-32', device=device)

# 2. Connect to Qdrant
client = QdrantClient(host = "localhost", port = 6333)
COLLECTION_NAME = "lumina_multimodal"

def search_lumina(text_query):
    print(f"\nSearching for: '{text_query}'....")

    #3. Encode the text query into a vector
    #CLIP maps this text into the same 512-D space as your images!
    query_vector = model.encode(text_query).tolist()

    #4. Perform the Search
    response = client.query_points(
        collection_name = COLLECTION_NAME,
        query = query_vector,
        limit = 3 #Return the top 3 closest matches
    )
    
    results = response.points

    #5. Display Results
    if not results:
        print("No matches found.")
        return
    
    print(f"{'Score':<10} | {'File Path':<30}")
    print("-" * 45)
    for res in results: 
        score = round(res.score, 4)
        path = res.payload.get("file_path", "Unknown")
        print(f"{score:<10} | {path:<30}")

if __name__ == "__main__":
    user_input = input("Enter search query: ")
    search_lumina(user_input)