from types import SimpleNamespace

from PIL import Image

from ingest_screenshots import discover_images
from meme_pipeline import build_point
from meme_retrieval import (
    _text_score,
    build_search_filter,
    hybrid_search,
    phash_distance,
    reverse_image_search,
    tokenize,
)


class StubEncoder:
    """Stands in for MemeEncoder so tests never download model weights."""

    def encode_image(self, image):
        return [0.1] * 512

    def encode_text(self, text):
        return [0.2] * 384


class StubClient:
    """Returns canned points per named vector without touching Qdrant."""

    def __init__(self, points_by_vector):
        self._points_by_vector = points_by_vector
        self.last_filter = None

    def query_points(self, collection_name=None, using=None, query_filter=None, **kwargs):
        self.last_filter = query_filter
        return SimpleNamespace(points=self._points_by_vector.get(using, []))


def hit(point_id, score, payload):
    return SimpleNamespace(id=point_id, score=score, payload=payload)


def test_exact_caption_scores_highest():
    payload = {"normalized_text": "me after deploying on friday", "template": "", "tags": []}
    assert _text_score("deploying on friday", payload) == 1.0


def test_term_overlap_is_partial_match():
    payload = {"normalized_text": "meeting could be an email", "template": "", "tags": []}
    assert 0 < _text_score("email work", payload) < 1


def test_punctuation_does_not_break_matching():
    payload = {"normalized_text": "me after deploying on friday.", "template": "", "tags": []}
    assert _text_score("deploying on friday!", payload) == 1.0


def test_substring_cannot_fake_a_match():
    payload = {"ocr_text": "category theory scatter plot", "template": "", "tags": []}
    assert _text_score("cat", payload) == 0.0


def test_tokenize_strips_case_and_punctuation():
    assert tokenize("Hello, World!!") == ["hello", "world"]


def test_safe_filter_uses_must_not_so_legacy_points_survive():
    search_filter = build_search_filter(safe_only=True)
    assert search_filter.must_not[0].key == "is_sensitive"
    assert not search_filter.must


def test_template_filter_is_lowercased_key():
    search_filter = build_search_filter(template="Drake", safe_only=False)
    assert search_filter.must[0].key == "template_key"
    assert search_filter.must[0].match.value == "drake"


def test_no_filters_returns_none():
    assert build_search_filter(template="", safe_only=False) is None


def test_phash_distance_basics():
    assert phash_distance("abcd1234", "abcd1234") == 0
    assert phash_distance("00", "01") == 1
    assert phash_distance("zz", "00") is None
    assert phash_distance("", "abcd") is None


def test_reverse_search_promotes_duplicates_over_higher_clip_scores():
    query_hash = "0" * 16
    far_neighbor = hit("far", 0.95, {"perceptual_hash": "f" * 16, "image_url": "http://x/f.png"})
    exact_repost = hit("dup", 0.60, {"perceptual_hash": query_hash, "image_url": "http://x/d.png"})
    client = StubClient({"visual": [far_neighbor, exact_repost]})
    results = reverse_image_search(client, "collection", [0.1] * 512, query_phash=query_hash)
    assert [r["id"] for r in results] == ["dup", "far"]
    assert results[0]["duplicate"] is True
    assert results[0]["phash_distance"] == 0
    assert results[1]["duplicate"] is False


def test_hybrid_search_merges_and_labels_matches():
    client = StubClient(
        {
            "visual": [hit("both", 0.8, {"title": "one", "tags": []}), hit("visual-only", 0.5, {"title": "two"})],
            "semantic_text": [hit("both", 0.6, {"title": "one", "tags": []})],
        }
    )
    results = hybrid_search(client, "collection", [0.1] * 512, [0.2] * 384, "anything")
    scores = {r["id"]: r for r in results}
    assert set(scores) == {"both", "visual-only"}
    assert "semantic" in scores["both"]["matched_on"]
    assert "visual" in scores["visual-only"]["matched_on"]
    assert scores["both"]["score"] > scores["visual-only"]["score"]


def test_build_point_payload_and_deterministic_id(tmp_path):
    image_path = tmp_path / "meme.png"
    Image.new("RGB", (8, 8), color=(200, 10, 10)).save(image_path)

    first = build_point(str(image_path), StubEncoder(), template="Drake", tags=["Fun"], caption="a caption", metadata={"is_sensitive": True})
    second = build_point(str(image_path), StubEncoder())

    assert first.id == second.id  # content-hash identity is stable
    payload = first.payload
    assert payload["template_key"] == "drake"
    assert payload["is_sensitive"] is True
    assert payload["caption"] == "a caption"
    assert payload["tags"] == ["fun"]
    assert len(payload["perceptual_hash"]) == 16
    assert len(first.vector["visual"]) == 512
    assert len(first.vector["semantic_text"]) == 384


def test_build_point_screenshot_mode(tmp_path):
    image_path = tmp_path / "shot.png"
    Image.new("RGB", (8, 8), color=(10, 200, 10)).save(image_path)

    point = build_point(
        str(image_path),
        StubEncoder(),
        tags=("screenshot",),
        media_type="screenshot",
        captured_at="2026-08-25T09:00:00+00:00",
    )

    assert point.payload["media_type"] == "screenshot"
    assert point.payload["created_at"] == "2026-08-25T09:00:00+00:00"
    assert point.payload["tags"] == ["screenshot"]
    # Memes keep the default ingestion timestamp behavior.
    default_point = build_point(str(image_path), StubEncoder())
    assert default_point.payload["media_type"] == "meme"
    assert default_point.payload["created_at"].endswith("+00:00")


def test_discover_images_filters_extensions_and_recursion(tmp_path):
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "b.PNG").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("skip me")
    subdirectory = tmp_path / "projects"
    subdirectory.mkdir()
    (subdirectory / "d.jpg").write_bytes(b"x")

    recursive = discover_images(tmp_path)
    flat = discover_images(tmp_path, recursive=False)

    assert recursive == sorted([str(tmp_path / "a.png"), str(tmp_path / "b.PNG"), str(subdirectory / "d.jpg")])
    assert str(subdirectory / "d.jpg") not in flat
    assert all(str(path).endswith((".png", ".PNG")) for path in flat)
