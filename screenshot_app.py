"""Private, local screenshot search. Images never leave your machine."""

import os

import streamlit as st
from PIL import Image
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from config import (
    CLIP_MODEL_NAME,
    PHASH_NEAR_DUPLICATE_DISTANCE,
    SCREENSHOT_COLLECTION_NAME,
    SCREENSHOT_ROOT,
    TEXT_MODEL_NAME,
)
from meme_pipeline import compute_phash, device_name
from meme_retrieval import hybrid_search, reverse_image_search

st.set_page_config(page_title="Lumina Screenshot Search", layout="wide", page_icon="🗂️")


def secret(name: str, fallback: str) -> str:
    try:
        return st.secrets.get(name, "") or fallback
    except Exception:
        return fallback


@st.cache_resource
def resources():
    url = secret("QDRANT_URL", os.getenv("QDRANT_URL", "http://localhost:6333"))
    key = secret("QDRANT_API_KEY", os.getenv("QDRANT_API_KEY", ""))
    client = QdrantClient(url=url, api_key=key) if key else QdrantClient(url=url)
    device = device_name()
    return (
        SentenceTransformer(CLIP_MODEL_NAME, device=device),
        SentenceTransformer(TEXT_MODEL_NAME, device=device),
        client,
    )


@st.cache_data(ttl=60)
def index_status() -> tuple[str, int]:
    try:
        client = resources()[2]
        if not client.collection_exists(SCREENSHOT_COLLECTION_NAME):
            return "missing", 0
        return "ready", client.count(collection_name=SCREENSHOT_COLLECTION_NAME, exact=True).count
    except Exception:
        return "unreachable", -1


try:
    visual_model, text_model, client = resources()
except Exception as exc:
    st.error(f"Could not start models/Qdrant: {exc}")
    st.stop()

status, count = index_status()
if status == "unreachable":
    st.error("Qdrant is unreachable — start it with `docker compose up -d` (or check QDRANT_URL).")
elif status == "missing":
    st.warning(
        f"Collection `{SCREENSHOT_COLLECTION_NAME}` does not exist yet. "
        "Run `python ingest_screenshots.py --limit 50` to index your first screenshots."
    )
elif count == 0:
    st.warning(
        "Collection exists but is empty — run `python ingest_screenshots.py` to index your screenshots."
    )

with st.sidebar:
    st.header("Library")
    if status == "ready":
        st.success(f"{count} screenshots indexed")
    safe_only = st.checkbox("Hide sensitive", True)
    st.caption(f"Folder: `{SCREENSHOT_ROOT}`\n\nRe-run `python ingest_screenshots.py` any time — only new files are processed.")

text_tab, image_tab = st.tabs(["🔎 Search", "🖼️ Find duplicate"])


def show_result(result: dict) -> None:
    file_path = result.get("file_path", "")
    if file_path and os.path.exists(file_path):
        st.image(file_path, use_container_width=True)
    else:
        st.warning("File moved or deleted — re-run ingestion to clean stale entries")
    label = result.get("caption") or result.get("ocr_text") or result.get("title") or "Screenshot"
    st.caption(label[:160])
    meta = f"score {result.get('score', 0):.3f} · {', '.join(result.get('matched_on', []))}"
    when = str(result.get("created_at", ""))[:10]
    if when:
        meta = f"{when} · {meta}"
    st.caption(meta)


with text_tab:
    query = st.text_input(
        "Describe what you remember",
        placeholder='e.g. "terminal error about permissions" or "hotel listing with brick wall"',
    )
    if query and status == "ready":
        try:
            results = hybrid_search(
                client,
                SCREENSHOT_COLLECTION_NAME,
                visual_model.encode(query, normalize_embeddings=True).tolist(),
                text_model.encode(query, normalize_embeddings=True).tolist(),
                query,
                limit=24,
                template=None,
                safe_only=safe_only,
            )
            if not results:
                st.info("Nothing matched. Try different wording — captions and on-screen text are both searched.")
            else:
                cols = st.columns(4)
                for index, result in enumerate(results):
                    with cols[index % 4]:
                        show_result(result)
        except Exception as exc:
            st.error(f"Search failed: {exc}")

with image_tab:
    st.caption("Upload a screenshot to check whether it is already in your library.")
    probe = st.file_uploader("Screenshot", type=("png", "jpg", "jpeg", "webp"))
    if probe and status == "ready":
        try:
            image = Image.open(probe).convert("RGB")
            results = reverse_image_search(
                client,
                SCREENSHOT_COLLECTION_NAME,
                visual_model.encode(image, normalize_embeddings=True).tolist(),
                query_phash=compute_phash(image),
                limit=12,
                safe_only=safe_only,
                near_duplicate_distance=PHASH_NEAR_DUPLICATE_DISTANCE,
            )
            if not results:
                st.info("No visually similar screenshots indexed yet.")
            else:
                cols = st.columns(4)
                for index, result in enumerate(results):
                    with cols[index % 4]:
                        distance = result.get("phash_distance")
                        if distance == 0:
                            st.success("Already in your library")
                        elif result.get("duplicate"):
                            st.info(f"Near duplicate · hash distance {distance}")
                        show_result(result)
        except Exception as exc:
            st.error(f"Image search failed: {exc}")

st.divider()
st.caption("Lumina Screenshot Search · local CLIP + MiniLM + SmolVLM captions + OCR · nothing uploaded")
