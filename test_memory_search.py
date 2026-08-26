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


def test_discover_images_includes_heic():
    from ingest_screenshots import DISCOVER_EXTENSIONS

    assert ".heic" in DISCOVER_EXTENSIONS and ".heif" in DISCOVER_EXTENSIONS


def test_detect_media_type_by_filename():
    from media_utils import detect_media_type

    assert detect_media_type("/x/Screenshot 2026-08-25 at 09.41.05.png") == "screenshot"
    assert detect_media_type("/x/screen shot 2026-01-01.jpg") == "screenshot"
    assert detect_media_type("/x/IMG_1234.HEIC") == "photo"
    assert detect_media_type("/x/vacation.png") == "photo"


def test_capture_datetime_prefers_exif_over_mtime(tmp_path):
    import os

    from media_utils import capture_datetime

    image_path = tmp_path / "IMG_0001.jpg"
    exif = Image.Exif()
    exif[0x9003] = "2020:01:15 08:30:00"
    Image.new("RGB", (8, 8)).save(image_path, exif=exif)
    os.utime(image_path, (1756200000, 1756200000))

    captured = capture_datetime(str(image_path))
    assert captured.startswith("2020-01-15")

    no_exif = tmp_path / "plain.png"
    Image.new("RGB", (8, 8)).save(no_exif)
    os.utime(no_exif, (1756100000, 1756100000))
    from datetime import datetime, timezone

    expected_day = datetime.fromtimestamp(1756100000, tz=timezone.utc).strftime("%Y-%m-%d")
    assert capture_datetime(str(no_exif)).startswith(expected_day)


def test_caption_prompt_varies_by_media_type():
    from captioner import caption_prompt

    screenshot_prompt = caption_prompt("screenshot")
    photo_prompt = caption_prompt("photo")
    meme_prompt = caption_prompt("meme")

    assert "app or website" in screenshot_prompt
    assert "main subject" in photo_prompt
    assert "meme" in meme_prompt
    assert all("Sensitive:" in p for p in (screenshot_prompt, photo_prompt, meme_prompt))
    # Unknown types fall back to the photo prompt rather than crashing.
    assert caption_prompt("hologram") == photo_prompt


# ---------- Phase 2: watch mode, stat cache, discovery hygiene ----------


def test_discover_images_skips_hidden_files_and_dirs(tmp_path):
    (tmp_path / "visible.png").write_bytes(b"x")
    (tmp_path / ".hidden.png").write_bytes(b"x")
    hidden_dir = tmp_path / ".Spotlight-V100"
    hidden_dir.mkdir()
    (hidden_dir / "index.png").write_bytes(b"x")

    assert discover_images(tmp_path) == [str(tmp_path / "visible.png")]


def test_stat_signature_tracks_mtime_and_size(tmp_path):
    import os

    from ingest_screenshots import stat_signature

    image_path = tmp_path / "a.png"
    Image.new("RGB", (4, 4)).save(image_path)
    first = stat_signature(os.stat(image_path))
    os.utime(image_path, (1000000000, 1000000000))
    second = stat_signature(os.stat(image_path))

    assert first != second
    assert stat_signature(os.stat(image_path)) == second  # stable when untouched


class FakeQdrant:
    """In-memory stand-in covering the client calls scan_library makes."""

    def __init__(self):
        self.points: dict[str, SimpleNamespace] = {}

    def retrieve(self, collection_name=None, ids=None, with_payload=True, **kwargs):
        return [self.points[str(i)] for i in ids if str(i) in self.points]

    def upsert(self, collection_name=None, points=None, wait=True, **kwargs):
        for point in points:
            self.points[str(point.id)] = point
        return SimpleNamespace(status="ok")

    def set_payload(self, collection_name=None, payload=None, points=None, wait=True, **kwargs):
        for point_id in points:
            if str(point_id) in self.points:
                self.points[str(point_id)].payload.update(payload)
        return SimpleNamespace(status="ok")

    def scroll(self, collection_name=None, limit=256, offset=None, with_payload=True, with_vectors=False, **kwargs):
        items = list(self.points.values())
        start = 0 if offset is None else len(items)  # single-page fake
        return items[start : start + limit], None if start + limit >= len(items) else start + limit

    def delete(self, collection_name=None, points_selector=None, wait=True, **kwargs):
        for point_id in points_selector.points:
            self.points.pop(str(point_id), None)
        return SimpleNamespace(status="ok")


def test_scan_library_cache_skips_unchanged_and_self_heals_after_touch(tmp_path):
    from ingest_screenshots import scan_library

    image_path = tmp_path / "Screenshot x.png"
    Image.new("RGB", (8, 8), color=(90, 90, 200)).save(image_path)

    state = {"version": 2}
    fake = FakeQdrant()
    common = dict(
        encoder=StubEncoder(),
        captioner=None,
        client=fake,
        state=state,
        do_sweep=False,
    )

    first = scan_library(tmp_path, "c", **common)
    assert first["indexed"] == 1 and first["unchanged"] == 0
    cached_entry = state["files"][str(image_path)]
    assert cached_entry["status"] == "indexed" and cached_entry["sig"]

    # Second pass: cache hit — nothing hashed, nothing sent to Qdrant.
    second = scan_library(tmp_path, "c", **common)
    assert second == {"indexed": 0, "unchanged": 1, "repaired": 0, "failed": 0}

    # Touch without content change: cache misses, Qdrant dedup catches it.
    import os

    os.utime(image_path, (1000000000, 1000000000))
    third = scan_library(tmp_path, "c", **common)
    assert third["unchanged"] == 1 and third["indexed"] == 0
    assert len(fake.points) == 1  # no duplicate point was created


def test_scan_library_flushes_pending_when_trailing_file_is_cache_hit(tmp_path):
    """Regression: a cache-hit on the last file must not strand earlier work."""
    import os

    from ingest_screenshots import scan_library

    first = tmp_path / "a.png"
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(first)
    second = tmp_path / "b.png"
    Image.new("RGB", (8, 8), color=(4, 5, 6)).save(second)

    state = {"version": 2}
    fake = FakeQdrant()
    common = dict(
        encoder=StubEncoder(),
        captioner=None,
        client=fake,
        state=state,
        do_sweep=False,
    )
    scan_library(tmp_path, "c", **common)
    assert len(fake.points) == 2

    # Overwrite `a` (sorts first) so the modified file is followed only by a
    # cache-hit file — previously the trailing hit skipped the final flush.
    Image.new("RGB", (14, 14), color=(9, 9, 9)).save(first)
    os.utime(second, (1000000000, 1000000000))  # keep second untouched

    result = scan_library(tmp_path, "c", **common)
    assert result["indexed"] == 1, f"modified file never reached Qdrant: {result}"
    assert len(fake.points) == 3


def test_debouncer_fires_once_after_quiet():
    import time

    from watcher import _Debouncer

    calls: list[str] = []
    debouncer = _Debouncer(lambda trigger: calls.append(trigger), delay=0.05)
    debouncer.notify("a")
    time.sleep(0.02)
    debouncer.notify("b")
    debouncer.notify("c")
    time.sleep(0.15)

    assert len(calls) == 1


def test_debouncer_cancel_prevents_callback():
    import time

    from watcher import _Debouncer

    calls: list[str] = []
    debouncer = _Debouncer(lambda trigger: calls.append(trigger), delay=0.05)
    debouncer.notify("a")
    debouncer.cancel()
    time.sleep(0.12)

    assert calls == []


def test_library_watcher_triggers_callback_on_new_file(tmp_path):
    import threading

    from watcher import LibraryWatcher

    seen = threading.Event()
    watcher = LibraryWatcher(
        [str(tmp_path)], lambda trigger: seen.set(), debounce_seconds=0.1
    )
    watcher.start()
    try:
        (tmp_path / "later.png").write_bytes(b"x")
        assert seen.wait(timeout=10), "watcher never fired"
        assert watcher.is_running
    finally:
        watcher.stop()
    assert not watcher.is_running


# ---------- Phase 3: NL dates, range filters, similar lookup, weights ----------


def _iso(day):
    from datetime import timezone

    return day.replace(tzinfo=timezone.utc)


def test_parse_date_range_yesterday():
    from datetime import datetime, timedelta, timezone

    from query_dates import parse_date_range

    cleaned, date_from, date_to = parse_date_range("terminal error yesterday please")
    now = datetime.now(timezone.utc)
    yest = (now - timedelta(days=1)).date()

    assert cleaned == "terminal error please"
    assert date_from.startswith(yest.isoformat())
    assert date_to.startswith(yest.isoformat())
    assert date_from.endswith("00:00:00+00:00")
    assert date_to.endswith("23:59:59.999999+00:00")


def test_parse_date_range_last_week_window():
    from datetime import datetime, timedelta, timezone

    from query_dates import parse_date_range

    _, date_from, date_to = parse_date_range("hotel brick wall last week")
    now = datetime.now(timezone.utc)

    assert date_from.startswith((now - timedelta(days=7)).date().isoformat())
    assert date_to.startswith((now - timedelta(days=1)).date().isoformat())


def test_parse_date_range_in_month_picks_most_recent():
    from datetime import datetime, timezone

    from query_dates import parse_date_range

    cleaned, date_from, date_to = parse_date_range("invoices in March")
    now = datetime.now(timezone.utc)
    year = now.year if now.month >= 3 else now.year - 1

    assert cleaned == "invoices"
    assert date_from == f"{year}-03-01T00:00:00+00:00"
    assert date_to.startswith(f"{year}-03-31T23:59:59")


def test_parse_date_range_since_month_runs_to_now():
    from datetime import datetime, timezone

    from query_dates import parse_date_range

    _, date_from, date_to = parse_date_range("since June screenshots of the bug")
    now = datetime.now(timezone.utc)
    year = now.year if now.month >= 6 else now.year - 1

    assert date_from == f"{year}-06-01T00:00:00+00:00"
    assert date_to.startswith(now.date().isoformat())


def test_parse_date_range_no_phrase_returns_query_untouched():
    from query_dates import parse_date_range

    cleaned, date_from, date_to = parse_date_range("cat sitting on keyboard")
    assert (cleaned, date_from, date_to) == ("cat sitting on keyboard", None, None)


def test_search_filter_carries_datetime_range():
    from datetime import datetime, timezone

    from meme_retrieval import build_search_filter

    search_filter = build_search_filter(
        safe_only=False,
        date_from="2026-03-01T00:00:00+00:00",
        date_to="2026-03-31T23:59:59+00:00",
    )
    condition = search_filter.must[0]

    assert condition.key == "created_at"
    assert condition.range.gte == datetime(2026, 3, 1, tzinfo=timezone.utc)
    assert condition.range.lte == datetime(2026, 3, 31, 23, 59, 59, tzinfo=timezone.utc)


class SimilarStubClient:
    def retrieve(self, collection_name=None, ids=None, with_vectors=False, **kwargs):
        return [SimpleNamespace(id="target", payload={}, vector={"visual": [0.5] * 512})]

    def query_points(self, **kwargs):
        return SimpleNamespace(
            points=[
                SimpleNamespace(id="target", score=0.99, payload={"caption": "the probe itself"}),
                SimpleNamespace(id="neighbor", score=0.91, payload={"caption": "lookalike"}),
            ]
        )


def test_find_similar_excludes_probe_and_keeps_order():
    from meme_retrieval import find_similar

    results = find_similar(SimilarStubClient(), "collection", "target", limit=5)

    assert [r["id"] for r in results] == ["neighbor"]
    assert results[0]["score"] == 0.91


def test_find_similar_missing_point_returns_empty():
    class EmptyClient:
        def retrieve(self, **kwargs):
            return []

    from meme_retrieval import find_similar

    assert find_similar(EmptyClient(), "collection", "ghost") == []


def test_hybrid_search_weights_can_flip_ranking():
    client = StubClient(
        {
            "visual": [hit("visual-heavy", 0.9, {"title": "one", "tags": []})],
            "semantic_text": [hit("semantic-heavy", 0.8, {"title": "two", "tags": []})],
        }
    )

    memory_first = hybrid_search(client, "c", [0.1] * 512, [0.2] * 384, "anything")
    flipped = hybrid_search(
        client, "c", [0.1] * 512, [0.2] * 384, "anything", weights=(1.0, 0.0, 0.0)
    )

    assert memory_first[0]["id"] == "semantic-heavy"  # 0.5 * 0.8 > 0.1 * 0.9
    assert flipped[0]["id"] == "visual-heavy"


def test_resolve_weights_defaults_to_memory_profile():
    from meme_retrieval import resolve_weights

    assert resolve_weights() == (0.10, 0.50, 0.40)
    assert resolve_weights("meme") == (0.20, 0.45, 0.35)
    assert resolve_weights("unknown-profile") == (0.10, 0.50, 0.40)
