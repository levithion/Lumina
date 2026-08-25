"""Standalone cloud Streamlit app; no local FastAPI process required."""

import streamlit as st
from PIL import Image
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from config import (
    CLIP_MODEL_NAME,
    MEME_COLLECTION_NAME,
    PHASH_NEAR_DUPLICATE_DISTANCE,
    QDRANT_API_KEY,
    QDRANT_URL,
    TEXT_MODEL_NAME,
)
from meme_pipeline import compute_phash, device_name
from meme_retrieval import hybrid_search, reverse_image_search

st.set_page_config(page_title="Lumina Meme Search", layout="wide", page_icon="😂")


def secret(name: str, fallback: str) -> str:
    """Read a Streamlit secret without crashing when no secrets file exists."""
    try:
        return st.secrets.get(name, "") or fallback
    except Exception:
        return fallback


@st.cache_resource
def resources():
    url = secret("QDRANT_URL", QDRANT_URL)
    key = secret("QDRANT_API_KEY", QDRANT_API_KEY)
    client = QdrantClient(url=url, api_key=key) if key else QdrantClient(url=url)
    device = device_name()
    return (
        SentenceTransformer(CLIP_MODEL_NAME, device=device),
        SentenceTransformer(TEXT_MODEL_NAME, device=device),
        client,
    )


@st.cache_data(ttl=60)
def indexed_count() -> int:
    try:
        return resources()[2].count(collection_name=MEME_COLLECTION_NAME, exact=True).count
    except Exception:
        return -1


try:
    visual_model, text_model, client = resources()
except Exception as exc:
    st.error(f"Cloud search is not configured: {exc}")
    st.stop()

count = indexed_count()
if count == 0:
    st.warning(
        f"Collection `{MEME_COLLECTION_NAME}` is empty. Run `python backfill_v2.py` to migrate "
        "the legacy index, or `python sync_memes.py --source meme_api` to ingest fresh memes."
    )
elif count < 0:
    st.error("Qdrant is unreachable — check QDRANT_URL / QDRANT_API_KEY.")

with st.sidebar:
    limit = st.slider("Results", 1, 50, 12)
    safe_only = st.checkbox("Safe content only", True)

text_tab, image_tab = st.tabs(["🔎 Text search", "🖼️ Image search"])

with text_tab:
    template = st.text_input("Template filter", placeholder="e.g. Drake")
    query = st.text_input("What meme are you looking for?", placeholder="e.g. programming failure")
    if query and count != 0:
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
                        label = result.get("caption") or result.get("ocr_text") or result.get("title") or "Meme"
                        st.caption(label[:140])
                        context = result.get("subreddit") or result.get("template")
                        meta = f"{result.get('score', 0):.3f} · {', '.join(result.get('matched_on', []))}"
                        if context:
                            meta = f"r/{context} · " + meta
                        st.caption(meta)
                        if result.get("source_url"):
                            st.link_button("Source", result["source_url"], use_container_width=True)
        except Exception as exc:
            st.error(f"Search failed: {exc}")

with image_tab:
    st.caption("Upload a meme to find its origin — exact reposts and near-duplicates rank first.")
    uploaded = st.file_uploader("Meme image", type=("png", "jpg", "jpeg", "webp", "gif"))
    camera = st.camera_input("…or snap one", help="Useful for screenshots of memes")
    probe = uploaded or camera
    if probe and count != 0:
        try:
            image = Image.open(probe).convert("RGB")
            results = reverse_image_search(
                client,
                MEME_COLLECTION_NAME,
                visual_model.encode(image, normalize_embeddings=True).tolist(),
                query_phash=compute_phash(image),
                limit=limit,
                safe_only=safe_only,
                near_duplicate_distance=PHASH_NEAR_DUPLICATE_DISTANCE,
            )
            if not results:
                st.info("No visually similar memes indexed yet.")
            else:
                cols = st.columns(4)
                for index, result in enumerate(results):
                    with cols[index % 4]:
                        source = result.get("image_url") or result.get("file_path")
                        if source:
                            st.image(source, use_container_width=True)
                        distance = result.get("phash_distance")
                        if distance == 0:
                            st.success("Exact repost detected")
                        elif result.get("duplicate"):
                            st.info(f"Near duplicate · hash distance {distance}")
                        label = result.get("caption") or result.get("ocr_text") or result.get("title") or "Meme"
                        st.caption(label[:140])
                        st.caption(f"CLIP similarity {result.get('score', 0):.3f}")
                        if result.get("source_url"):
                            st.link_button("Source post", result["source_url"], use_container_width=True)
        except Exception as exc:
            st.error(f"Image search failed: {exc}")

st.divider()
st.caption("Lumina Meme Search · CLIP + MiniLM hybrid retrieval · SmolVLM captions · pHash repost detection")
