import streamlit as st
import requests
import os

# 1. Page Configuration
st.set_page_config(page_title="Lumina Search", layout="wide", page_icon="🔍")

# UI Styling
st.title("🔍 Lumina: Multimodal Search Engine")
st.markdown("---")

# 2. Sidebar & Connection Verification
st.sidebar.header("System Status")
search_url = "http://localhost:8000/search"
health_url = "http://localhost:8000/health"

# Real-time backend status check
try:
    health_resp = requests.get(health_url, timeout=2)
    if health_resp.status_code == 200:
        st.sidebar.success("✅ Backend: Online")
        st.sidebar.info(f"Device: {health_resp.json().get('device', 'unknown')}")
    else:
        st.sidebar.error("❌ Backend: Error")
except:
    st.sidebar.error("⚠️ Backend: Offline")

top_k = st.sidebar.slider("Number of results", 1, 20, 9)

# 3. Search Interface
query = st.text_input("Describe what you're looking for...", placeholder="e.g. a tall building at sunset")

if query:
    with st.spinner(f"Searching 8,000+ images for '{query}'..."):
        try:
            payload = {"query": query, "limit": top_k}
            # Added a 10-second timeout for cloud latency
            response = requests.post(search_url, json=payload, timeout=10)
            results = response.json()

            if results:
                st.subheader(f"Top matches for: {query}")
                
                # Grid layout: 3 images per row
                cols = st.columns(3) 
                for idx, hit in enumerate(results):
                    with cols[idx % 3]:
                        image_path = hit["file_path"]
                        score = hit["score"]
                        
                        # IMPORTANT: In the cloud, the physical file might not exist yet
                        if os.path.exists(image_path):
                            st.image(image_path, use_container_width=True, 
                                     caption=f"Similarity: {score}")
                        else:
                            # If file is missing, we show the path and score
                            st.info(f"Match Found! Score: {score}")
                            st.caption(f"Path: {image_path}")
                            st.warning("🖼️ Image file not hosted in cloud container.")
            else:
                st.info("No matching results found in the vector database.")
                
        except Exception as e:
            st.error(f"Search failed. Please ensure the FastAPI backend is running. Error: {e}")

st.markdown("---")
st.caption("Lumina Search Engine | Built with FastAPI, Qdrant & CLIP")