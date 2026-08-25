import os

import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Lumina Meme Search", layout="wide", page_icon="😂")
st.title("😂 Lumina Meme Search")
st.caption("Search memes by caption, meaning, reaction, or template — or find a meme from an image.")

search_url = os.getenv("SEARCH_API_URL", "http://localhost:8000/search")
image_search_url = os.getenv("IMAGE_SEARCH_API_URL", "http://localhost:8000/search/image")
health_url = os.getenv("HEALTH_API_URL", "http://localhost:8000/health")

with st.sidebar:
    st.header("Search options")
    top_k = st.slider("Results", 1, 50, 12)
    safe_only = st.checkbox("Safe content only", value=True)
    try:
        health = requests.get(health_url, timeout=2).json()
        if health.get("status") == "online":
            st.success(f"API online · {health.get('indexed_memes', 0)} memes")
        else:
            st.warning(f"API degraded · database {health.get('database', 'unreachable')}")
    except requests.RequestException:
        st.error("API offline")


def render_grid(results, empty_label):
    cols = st.columns(4)
    for index, result in enumerate(results):
        with cols[index % 4]:
            source = result.get("image_url") or result.get("file_path")
            if source and (source.startswith("http") or os.path.exists(source)):
                st.image(source, use_container_width=True)
            else:
                st.warning("Image is not publicly available")
            distance = result.get("phash_distance")
            if distance == 0:
                st.success("Exact repost detected")
            elif result.get("duplicate"):
                st.info(f"Near duplicate · hash distance {distance}")
            label = result.get("caption") or result.get("ocr_text") or result.get("title") or empty_label
            st.caption(label[:140])
            meta = f"Score {result.get('score', 0):.3f} · {', '.join(result.get('matched_on', []))}"
            if result.get("subreddit"):
                meta = f"r/{result['subreddit']} · " + meta
            st.caption(meta)


text_tab, image_tab = st.tabs(["🔎 Text search", "🖼️ Image search"])

with text_tab:
    template = st.text_input("Template filter", placeholder="e.g. Drake")
    query = st.text_input("What meme are you looking for?", placeholder="e.g. when production breaks on Friday")
    if query:
        try:
            with st.spinner("Searching memes..."):
                response = requests.post(
                    search_url,
                    json={"query": query, "limit": top_k, "template": template or None, "safe_only": safe_only},
                    timeout=30,
                )
                response.raise_for_status()
                results = response.json()
            if not results:
                st.info("No matching memes found. Try a caption, reaction, or template name.")
            else:
                st.subheader(f"Matches for “{query}”")
                render_grid(results, "Meme match")
        except requests.RequestException as exc:
            st.error(f"Search request failed: {exc}")

with image_tab:
    st.caption("Upload a meme to find its origin in the index.")
    uploaded = st.file_uploader("Meme image", type=("png", "jpg", "jpeg", "webp", "gif"))
    if uploaded:
        try:
            with st.spinner("Searching visually..."):
                response = requests.post(
                    image_search_url,
                    files={"file": (uploaded.name, uploaded.getvalue())},
                    params={"limit": top_k, "safe_only": str(safe_only).lower()},
                    timeout=60,
                )
                response.raise_for_status()
                results = response.json()
            if not results:
                st.info("No visually similar memes indexed yet.")
            else:
                Image.open(uploaded).thumbnail((320, 320))
                st.image(Image.open(uploaded), caption="Your upload", width=200)
                st.subheader("Visual matches")
                render_grid(results, "Meme match")
        except requests.RequestException as exc:
            st.error(f"Image search request failed: {exc}")

st.divider()
st.caption("Lumina Meme Search · OCR + CLIP + SmolVLM captions + pHash repost detection")
