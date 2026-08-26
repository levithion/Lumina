"""Lumina memory-search server: one process hosts models, DB, and watcher.

Embedded Qdrant allows a single process per storage folder, so this server
owns everything: CLIP/MiniLM encoding, retrieval, image serving, duplicate
checks, optional FSEvents watching, and manual reindexes. Run with:

    python server.py [--port 8337] [--watch] [--folder ~/Screenshots]
"""

from __future__ import annotations

import argparse
import io
import mimetypes
import subprocess
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from qdrant_client.models import FieldCondition, Filter, MatchValue

from config import (
    MAX_DOWNLOAD_BYTES,
    MEMORY_COLLECTION_NAME,
    MEMORY_STATE_PATH,
    SCREENSHOT_ROOT,
    SERVER_HOST,
    SERVER_PORT,
    PHASH_NEAR_DUPLICATE_DISTANCE,
    qdrant_client,
)
from init_db import initialize_database
from pipeline import ImageEncoder, compute_phash
from retrieval import find_similar, hybrid_search, reverse_image_search
from query_dates import parse_date_range

PLACEHOLDER_HTML = """<!doctype html><html><head><title>Lumina</title></head>
<body style="font-family:-apple-system,sans-serif;background:#0e1116;color:#e6e6e6;
display:grid;place-items:center;height:100vh;margin:0">
<div style="text-align:center">
<h1>Lumina</h1><p>Private memory search · API live on :{port}</p>
<p style="opacity:.6">ui/ folder not found — endpoints under <code>/api/…</code></p>
</div></body></html>"""

UI_DIR = Path(__file__).resolve().parent / "ui"


class OpenRequest(BaseModel):
    path: str


def create_app(
    *,
    encoder: Any = None,
    client: Any = None,
    collection: str = "",
    folder: Path | None = None,
    watch: bool | None = None,
) -> FastAPI:
    """Build the app. Injecting encoder/client skips resource loading (tests)."""
    state: dict[str, Any] = {
        "encoder": encoder,
        "captioner": False,  # loaded lazily alongside the encoder
        "client": client,
        "collection": collection or MEMORY_COLLECTION_NAME,
        "folder": folder or Path(SCREENSHOT_ROOT),
        "watch": bool(watch),
        "ingest_lock": threading.Lock(),
        "reindex_running": False,
        "watcher": None,
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        if state["client"] is None:
            initialize_database(state["collection"])
            state["client"] = qdrant_client()
        if state["encoder"] is None:
            state["encoder"] = ImageEncoder()
            try:
                from captioner import build_captioner

                state["captioner"] = build_captioner()
            except Exception as exc:
                print(f"Captioner unavailable ({exc}); reindex runs OCR+visual only")
        if state["watch"]:
            _start_watcher(state)
        yield
        watcher = state.get("watcher")
        if watcher is not None:
            watcher.stop()

    app = FastAPI(title="Lumina", version="4.0", lifespan=lifespan)
    if UI_DIR.is_dir():
        app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")

    # ---------- helpers ----------

    def _models() -> tuple[Any, Any]:
        if state["encoder"] is None or state["client"] is None:
            raise HTTPException(503, "server still loading models/database")
        return state["encoder"], state["client"]

    def _require_indexed_path(raw_path: str) -> Path:
        """Only files actually present in the index may be read or opened."""
        _, client = _models()
        path = Path(raw_path)
        if not path.is_absolute():
            raise HTTPException(400, "absolute path required")
        resolved = path.resolve()
        hits, _ = client.scroll(
            collection_name=state["collection"],
            scroll_filter=Filter(
                must=[FieldCondition(key="local_path", match=MatchValue(value=str(resolved)))]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        if not hits:
            raise HTTPException(404, "path not indexed")
        if not resolved.is_file():
            raise HTTPException(410, "file has vanished; re-run reindex")
        return resolved

    def _scan_once(trigger: str) -> dict[str, int]:
        """One incremental library pass; serialized with the watcher."""
        from ingest_screenshots import (
            discover_images,
            load_state,
            save_state,
            scan_library,
        )

        with state["ingest_lock"]:
            state_path = Path(MEMORY_STATE_PATH)
            lib_state = load_state(state_path)
            stats = scan_library(
                state["folder"],
                state["collection"],
                encoder=state["encoder"],
                captioner=state["captioner"] or None,
                client=state["client"],
                state=lib_state,
                state_path=state_path,
                do_sweep=True,
            )
            print(f"[{trigger}] indexed {stats['indexed']} unchanged {stats['unchanged']}")
            return stats

    def _start_watcher(st: dict[str, Any]) -> None:
        from watcher import LibraryWatcher

        def on_change(_trigger: str) -> None:
            try:
                _scan_once("watch")
            except Exception as exc:
                print(f"WARNING: watch scan failed: {exc}")

        def initial_scan() -> None:
            try:
                _scan_once("startup")
            except Exception as exc:
                print(f"WARNING: startup scan failed: {exc}")
            watcher_obj = LibraryWatcher([str(st["folder"])], on_change, debounce_seconds=10.0)
            watcher_obj.start()
            st["watcher"] = watcher_obj
            print(f"Watching {st['folder']}")

        threading.Thread(target=initial_scan, name="lumina-initial-scan", daemon=True).start()

    # ---------- routes ----------

    @app.get("/", response_class=HTMLResponse)
    def root() -> Any:
        if UI_DIR.is_dir():
            return RedirectResponse("/ui/index.html")
        return HTMLResponse(PLACEHOLDER_HTML.format(port=SERVER_PORT))

    @app.get("/api/search")
    def api_search(
        q: str,
        limit: int = 24,
        safe: bool = True,
    ) -> JSONResponse:
        encoder_obj, client_obj = _models()
        cleaned, date_from, date_to = parse_date_range(q)
        results = hybrid_search(
            client_obj,
            state["collection"],
            encoder_obj.visual.encode(cleaned, normalize_embeddings=True).tolist(),
            encoder_obj.text.encode(cleaned, normalize_embeddings=True).tolist(),
            cleaned,
            limit=max(1, min(limit, 96)),
            template=None,
            safe_only=safe,
            date_from=date_from,
            date_to=date_to,
        )
        return JSONResponse(
            {
                "query": cleaned,
                "date_filter": [date_from, date_to] if date_from else None,
                "results": results,
            }
        )

    @app.get("/api/similar/{point_id}")
    def api_similar(point_id: str, limit: int = 12, safe: bool = True) -> JSONResponse:
        _, client_obj = _models()
        results = find_similar(
            client_obj,
            state["collection"],
            point_id,
            limit=max(1, min(limit, 48)),
            safe_only=safe,
        )
        return JSONResponse({"results": results})

    @app.post("/api/duplicate")
    async def api_duplicate(probe: UploadFile = File(...), safe: bool = True) -> JSONResponse:
        encoder_obj, client_obj = _models()
        payload = await probe.read()
        if len(payload) > MAX_DOWNLOAD_BYTES:
            raise HTTPException(413, "file exceeds MAX_DOWNLOAD_BYTES")
        from PIL import Image
        from media_utils import ensure_heif_support

        ensure_heif_support()
        try:
            image = Image.open(io.BytesIO(payload)).convert("RGB")
        except Exception:
            raise HTTPException(400, "not a readable image") from None
        results = reverse_image_search(
            client_obj,
            state["collection"],
            encoder_obj.encode_image(image),
            query_phash=compute_phash(image),
            limit=12,
            safe_only=safe,
            near_duplicate_distance=PHASH_NEAR_DUPLICATE_DISTANCE,
        )
        duplicates = [r for r in results if r.get("duplicate")]
        return JSONResponse(
            {"duplicate": bool(duplicates), "results": results[:6]}
        )

    @app.get("/api/image")
    def api_image(path: str) -> FileResponse:
        resolved = _require_indexed_path(path)
        media_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
        return FileResponse(resolved, media_type=media_type)

    @app.post("/api/open")
    def api_open(body: OpenRequest) -> JSONResponse:
        resolved = _require_indexed_path(body.path)
        completed = subprocess.run(
            ["open", "-R", str(resolved)], capture_output=True, text=True, timeout=15
        )
        if completed.returncode != 0:
            raise HTTPException(500, completed.stderr.strip() or "Finder reveal failed")
        return JSONResponse({"opened": str(resolved)})

    @app.get("/api/status")
    def api_status() -> JSONResponse:
        from ingest_screenshots import load_state

        _, client_obj = _models()
        count = 0
        try:
            count = client_obj.count(collection_name=state["collection"], exact=True).count
        except Exception:
            count = -1
        lib_state = load_state(Path(MEMORY_STATE_PATH))
        return JSONResponse(
            {
                "points": count,
                "collection": state["collection"],
                "folders": [str(state["folder"])],
                "last_scan": lib_state.get("last_scan"),
                "failures": len(lib_state.get("failures", {})),
                "watching": bool(state.get("watcher") and state["watcher"].is_running),
                "reindex_running": state["reindex_running"],
            }
        )

    @app.post("/api/reindex")
    def api_reindex() -> JSONResponse:
        if state["reindex_running"]:
            return JSONResponse({"started": False, "reason": "already running"})
        state["reindex_running"] = True

        def job() -> None:
            try:
                _scan_once("manual-reindex")
            finally:
                state["reindex_running"] = False

        threading.Thread(target=job, name="lumina-reindex", daemon=True).start()
        return JSONResponse({"started": True})

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=SERVER_HOST)
    parser.add_argument("--port", type=int, default=SERVER_PORT)
    parser.add_argument("--folder", default=SCREENSHOT_ROOT)
    parser.add_argument("--collection", default=MEMORY_COLLECTION_NAME)
    parser.add_argument("--watch", action="store_true", help="also watch the folder for changes")
    args = parser.parse_args()

    global app
    app = create_app(folder=Path(args.folder).expanduser().resolve(), watch=args.watch)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
