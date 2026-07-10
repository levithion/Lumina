import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

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

html = "<html><body>"
for hit in response.points:
    path = hit.payload.get("file_path", "")
    score = hit.score
    html += f"<div><p>{path} - Score: {score}</p><img src='{path}' width='200'/></div>"
html += "</body></html>"
with open("test_results.html", "w") as f:
    f.write(html)
