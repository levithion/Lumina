"""Standalone cloud Streamlit app; no local FastAPI process required."""

import os

import streamlit as st
import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from config import CLIP_MODEL_NAME, MEME_COLLECTION_NAME, QDRANT_API_KEY, QDRANT_URL, TEXT_MODEL_NAME
from meme_pipeline import device_name
from meme_retrieval import hybrid_search

st.set_page_config(page_title="Lumina Meme Search", layout="wide", page_icon="😂")


@st.cache_resource
def resources():
    url = st.secrets.get("QDRANT_URL", QDRANT_URL)
    key = st.secrets.get("QDRANT_API_KEY", QDRANT_API_KEY)
    client = QdrantClient(url=url, api_key=key) if key else QdrantClient(url=url)
    device = device_name()
    return (
        SentenceTransformer(CLIP_MODEL_NAME, device=device),
        SentenceTransformer(TEXT_MODEL_NAME, device=device),
        client,
    )


st.title("😂 Lumina Meme Search")
st.caption("Search current memes by caption, meaning, reaction, or subreddit.")
try:
    visual_model, text_model, client = resources()
except Exception as exc:
    st.error(f"Cloud search is not configured: {exc}")
    st.stop()

with st.sidebar:
    limit = st.slider("Results", 1, 50, 12)
    template = st.text_input("Template filter")
    safe_only = st.checkbox("Safe content only", True)

query = st.text_input("What meme are you looking for?", placeholder="e.g. programming failure")
if query:
    try:
        results = hybrid_search(
            client,
            MEME_COLLECTION_NAME,
            visual_model.encode(query, normalize_embeddings=True).tolist(),
            text_model.encode(query, normalize_embeddings=True).tolist(),
            query,
            limit,
            template or None,
            safe_only,
        )
        if not results:
            st.info("No matching memes found.")
        else:
            cols = st.columns(4)
            for index, result in enumerate(results):
                with cols[index % 4]:
                    source = result.get("image_url") or result.get("file_path")
                    if source:
                        st.image(source, use_container_width=True)
                    st.caption((result.get("ocr_text") or result.get("title") or "Meme")[:140])
                    st.caption(f"{result.get('score', 0):.3f} · {', '.join(result.get('matched_on', []))}")
    except Exception as exc:
        st.error(f"Search failed: {exc}")

st.divider()
st.caption("Lumina Meme Search")
