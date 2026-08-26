"""API tests for server.py using injected stubs — no models, no real Qdrant."""

import io
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

# Configure embedded storage paths BEFORE any project import reads them.
_TMP = Path(tempfile.mkdtemp(prefix="lumina_api_"))
os.environ.setdefault("QDRANT_URL", "")
os.environ["MEMORY_STORAGE_PATH"] = str(_TMP / "qd")
os.environ["MEMORY_STATE_PATH"] = str(_TMP / "state.json")
os.environ["CAPTION_ENABLED"] = "0"

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

import server as server_module  # noqa: E402
from pipeline import compute_phash  # noqa: E402


class Vec(list):
    """Mimics numpy arrays just enough for server code calling .tolist()."""

    def tolist(self):
        return list(self)


class StubEncoder:
    """Matches the server's encoder surface: .visual/.text SentenceTransformers."""

    class _Side:
        def __init__(self, is_visual):
            self.is_visual = is_visual

        def encode(self, text, normalize_embeddings=False):
            return Vec([0.1] * 512) if self.is_visual else Vec([0.2] * 384)

    def __init__(self):
        self.visual = self._Side(is_visual=True)
        self.text = self._Side(is_visual=False)

    def encode_image(self, image):
        return Vec([0.3] * 512)


class FakeServerClient:
    """Covers every client call the API routes can make."""

    def __init__(self):
        self.points = {}  # id -> payload dict
        self.opened = None

    # --- retrieval ---
    def query_points(self, collection_name=None, query=None, using=None, limit=None,
                     with_payload=True, query_filter=None, **kwargs):
        hits = [
            SimpleNamespace(id=pid, score=score, payload=dict(payload))
            for pid, (score, payload) in sorted(
                self.points.items(), key=lambda kv: kv[1][0], reverse=True
            )
        ]
        return SimpleNamespace(points=hits[:limit])

    def retrieve(self, collection_name=None, ids=None, with_payload=True, with_vectors=False, **kwargs):
        out = []
        for pid in ids or []:
            if str(pid) in self.points:
                vector = {"visual": [0.5] * 512} if with_vectors else None
                score_payload = self.points[str(pid)][1]
                out.append(SimpleNamespace(id=str(pid), payload=dict(score_payload), vector=vector))
        return out

    def scroll(self, collection_name=None, scroll_filter=None, limit=256, offset=None,
               with_payload=True, with_vectors=False, **kwargs):
        if not scroll_filter:
            return [], None  # end-of-collection for the stale sweeper
        if isinstance(scroll_filter, dict):
            value = scroll_filter["must"][0]["match"]["value"]
        else:
            value = scroll_filter.must[0].match.value
        matches = [
            SimpleNamespace(id=pid, payload=dict(payload), vector=None)
            for pid, (_, payload) in self.points.items()
            if payload.get("local_path") == value
        ]
        return matches[:limit], None

    def count(self, collection_name=None, exact=True, **kwargs):
        return SimpleNamespace(count=len(self.points))

    # --- ingestion-side (used by reindex/sweep paths) ---
    def upsert(self, collection_name=None, points=None, wait=True, **kwargs):
        for point in points or []:
            self.points[str(point.id)] = (0.5, dict(point.payload))
        return SimpleNamespace(status="ok")

    def set_payload(self, **kwargs):
        return SimpleNamespace(status="ok")

    def delete(self, collection_name=None, points_selector=None, wait=True, **kwargs):
        return SimpleNamespace(status="ok")


def seeded_client(image_path: Path) -> FakeServerClient:
    client = FakeServerClient()
    with Image.open(image_path) as img:
        digest = compute_phash(img.convert("RGB"))
    client.points["point-1"] = (
        0.9,
        {
            "local_path": str(image_path),
            "caption": "terminal error about permissions",
            "ocr_text": "",
            "created_at": "2026-08-25T09:00:00+00:00",
            "media_type": "screenshot",
            "perceptual_hash": digest,
            "tags": [],
            "template": "",
            "title": "",
            "subreddit": "",
            "image_url": "",
            "source_url": "",
        },
    )
    return client


def make_client(tmp_path: Path):
    image_path = tmp_path / "shot.png"
    Image.new("RGB", (32, 32), color=(10, 20, 30)).save(image_path)
    app = server_module.create_app(
        encoder=StubEncoder(), client=seeded_client(image_path), folder=tmp_path
    )
    return TestClient(app), image_path


def test_search_returns_results_and_clean_query(tmp_path):
    http, _ = make_client(tmp_path)
    with http:
        response = http.get("/api/search", params={"q": "terminal error permissions"})
        assert response.status_code == 200
        body = response.json()
        assert body["query"] == "terminal error permissions"
        assert body["date_filter"] is None
        assert body["results"][0]["id"] == "point-1"
        assert "semantic" in body["results"][0]["matched_on"]


def test_search_parses_date_phrase_into_filter(tmp_path):
    http, _ = make_client(tmp_path)
    with http:
        body = http.get("/api/search", params={"q": "error yesterday"}).json()

    assert body["query"] == "error"
    assert body["date_filter"] is not None
    date_from, date_to = body["date_filter"]
    assert date_from < date_to


def test_similar_excludes_probe_point(tmp_path):
    http, _ = make_client(tmp_path)
    with http:
        body = http.get("/api/similar/point-1").json()
    assert body["results"] == []  # only indexed point IS the probe


def test_duplicate_flags_exact_upload(tmp_path):
    http, image_path = make_client(tmp_path)
    with http:
        response = http.post(
            "/api/duplicate",
            files={"probe": ("shot.png", image_path.read_bytes(), "image/png")},
        )
        assert response.status_code == 200
        body = response.json()
    assert body["duplicate"] is True
    assert body["results"][0]["phash_distance"] == 0


def test_image_serves_only_indexed_paths(tmp_path):
    http, image_path = make_client(tmp_path)
    stranger = tmp_path / "stranger.png"
    Image.new("RGB", (8, 8)).save(stranger)

    with http:
        ok = http.get("/api/image", params={"path": str(image_path)})
        missing = http.get("/api/image", params={"path": str(stranger)})
        relative = http.get("/api/image", params={"path": "shot.png"})

    assert ok.status_code == 200 and ok.content.startswith(b"\x89PNG")
    assert missing.status_code == 404
    assert relative.status_code == 400


def test_open_reveals_indexed_file_in_finder(tmp_path, monkeypatch):
    http, image_path = make_client(tmp_path)

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(server_module.subprocess, "run", fake_run)
    with http:
        response = http.post("/api/open", json={"path": str(image_path)})

    assert response.status_code == 200
    assert captured["cmd"] == ["open", "-R", str(image_path)]


def test_status_reports_library_state(tmp_path):
    http, _ = make_client(tmp_path)
    with http:
        body = http.get("/api/status").json()
    assert body["points"] == 1
    assert body["watching"] is False
    assert body["reindex_running"] is False
    assert body["collection"] == server_module.MEMORY_COLLECTION_NAME


def test_reindex_completes_on_empty_folder(tmp_path):
    empty = tmp_path / "empty-library"
    empty.mkdir()
    app = server_module.create_app(encoder=StubEncoder(), client=FakeServerClient(), folder=empty)
    with TestClient(app) as http:
        started = http.post("/api/reindex").json()
        assert started["started"] is True
        import time

        deadline = time.time() + 10
        while time.time() < deadline:
            if not http.get("/api/status").json()["reindex_running"]:
                break
            time.sleep(0.1)
        assert http.get("/api/status").json()["reindex_running"] is False


# ---------- Phase 5: static UI ----------


def test_root_redirects_to_ui():
    app = server_module.create_app(encoder=StubEncoder(), client=FakeServerClient())
    with TestClient(app) as http:
        response = http.get("/", follow_redirects=False)
    assert response.status_code in (301, 307)
    assert response.headers["location"] == "/ui/index.html"


def test_ui_assets_served():
    app = server_module.create_app(encoder=StubEncoder(), client=FakeServerClient())
    with TestClient(app) as http:
        page = http.get("/ui/index.html")
        styles = http.get("/ui/style.css")
        script = http.get("/ui/app.js")

    assert page.status_code == 200 and b"Lumina" in page.content
    assert styles.status_code == 200
    assert script.status_code == 200 and b"doSearch" in script.content
