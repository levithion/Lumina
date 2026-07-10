import streamlit as st
import os
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# 1. Page Configuration
st.set_page_config(page_title="Lumina Search", layout="wide", page_icon="🔍")

# UI Styling
st.title("🔍 Lumina: Multimodal Search Engine")
st.markdown("---")

# 2. Setup caching for heavy models and clients
# This prevents the model from reloading every time you click a button!
@st.cache_resource
def load_model_and_client():
    # Streamlit Cloud uses st.secrets to store sensitive variables securely
    qdrant_url = os.environ.get("QDRANT_URL", "https://1f7c7f33-b52f-4965-aa17-afb87273d512.us-east-2-0.aws.cloud.qdrant.io")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6NTgxYzBjNzUtMmY5MC00YzExLWI0NDYtNmFmY2IxMzVmNTMxIn0.jWEKq72tOwZ88F5GdiQuUZU8aDNKDNyhUPBGUZKfsX4")
    
    try:
        if "QDRANT_URL" in st.secrets:
            qdrant_url = st.secrets["QDRANT_URL"]
        if "QDRANT_API_KEY" in st.secrets:
            qdrant_api_key = st.secrets["QDRANT_API_KEY"]
    except Exception:
        # Fallback to hardcoded variables if testing locally without secrets.toml
        pass
    
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = SentenceTransformer('clip-ViT-B-32', device=device)
    
    if qdrant_api_key:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    else:
        client = QdrantClient(url=qdrant_url)
        
    return model, client

st.sidebar.header("System Status")

try:
    with st.spinner("Loading AI model and connecting to database..."):
        model, client = load_model_and_client()
    st.sidebar.success("✅ System: Online")
    st.sidebar.info(f"Device: {model.device}")
except Exception as e:
    st.sidebar.error("❌ System: Error loading model or database")
    st.sidebar.error(str(e))
    st.stop()

COLLECTION_NAME = "lumina_multimodal"
top_k = st.sidebar.slider("Number of results", 1, 20, 9)

# 3. Search Interface
query = st.text_input("Describe what you're looking for...", placeholder="e.g. a tall building at sunset")

if query:
    with st.spinner(f"Searching for '{query}'..."):
        try:
            # Encode text query directly within Streamlit
            vector = model.encode(query).tolist()

            # Query Qdrant Cloud
            response = client.query_points(
                collection_name=COLLECTION_NAME,
                query=vector,
                limit=top_k
            )
            
            if response.points:
                st.subheader(f"Top matches for: {query}")
                
                # Grid layout: 3 images per row
                cols = st.columns(3) 
                for idx, hit in enumerate(response.points):
                    with cols[idx % 3]:
                        image_path = hit.payload.get("file_path", "Unknown")
                        score = round(hit.score, 4)
                        
                        if os.path.exists(image_path):
                            st.image(image_path, use_container_width=True, 
                                     caption=f"Similarity: {score}")
                        else:
                            st.info(f"Match Found! Score: {score}")
                            st.caption(f"Path: {image_path}")
                            st.warning("🖼️ Image file not hosted in this cloud container.")
            else:
                st.info("No matching results found in the vector database.")
                
        except Exception as e:
            st.error(f"Search failed. Error: {e}")

st.markdown("---")
st.caption("Lumina Search Engine | Built with Streamlit, Qdrant & CLIP")
