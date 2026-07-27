import os

import requests
import streamlit as st

st.set_page_config(page_title="Lumina Meme Search", layout="wide", page_icon="😂")
st.title("😂 Lumina Meme Search")
st.caption("Search memes by caption, meaning, reaction, or template.")

search_url = os.getenv("SEARCH_API_URL", "http://localhost:8000/search")
health_url = os.getenv("HEALTH_API_URL", "http://localhost:8000/health")

with st.sidebar:
    st.header("Search options")
    top_k = st.slider("Results", 1, 50, 12)
    template = st.text_input("Template filter", placeholder="e.g. Drake")
    safe_only = st.checkbox("Safe content only", value=True)
    try:
        health = requests.get(health_url, timeout=2).json()
        st.success(f"API online · {health.get('indexed_memes', 0)} memes")
    except requests.RequestException:
        st.error("API offline")

query = st.text_input("What meme are you looking for?", placeholder="e.g. when production breaks on Friday")
if query:
    try:
        with st.spinner("Searching memes..."):
            response = requests.post(search_url, json={"query": query, "limit": top_k, "template": template or None, "safe_only": safe_only}, timeout=30)
            response.raise_for_status()
            results = response.json()
        if not results:
            st.info("No matching memes found. Try a caption, reaction, or template name.")
        else:
            st.subheader(f"Matches for “{query}”")
            cols = st.columns(4)
            for index, result in enumerate(results):
                with cols[index % 4]:
                    source = result.get("image_url") or result.get("file_path")
                    if source and (source.startswith("http") or os.path.exists(source)):
                        st.image(source, use_container_width=True)
                    else:
                        st.warning("Image is not publicly available")
                    caption = result.get("ocr_text") or result.get("template") or "Meme match"
                    st.caption(caption[:140])
                    st.caption(f"Score {result.get('score', 0):.3f} · {', '.join(result.get('matched_on', []))}")
    except requests.RequestException as exc:
        st.error(f"Search request failed: {exc}")

st.divider()
st.caption("Lumina Meme Search · OCR + CLIP + semantic retrieval")
