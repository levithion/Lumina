"""Index a local screenshots/photos library for private semantic search.

Everything stays on your machine: images are never uploaded, embeddings and
VLM captions are computed locally, and points reference absolute file paths.
Content-hash IDs make reruns free — unchanged files are skipped via a stat
signature cache (no re-hashing), moved files are repaired in place, and
deleted/superseded versions are swept. ``--watch`` keeps the index fresh by
scanning after filesystem activity settles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qdrant_client.models import PointIdsList

from config import (
    MEMORY_COLLECTION_NAME,
    MEMORY_STATE_PATH,
    SCREENSHOT_ROOT,
    qdrant_client,
)
from init_meme_db import initialize_meme_database
from media_utils import PHOTO_EXTENSIONS, capture_datetime, detect_media_type, open_image
from meme_pipeline import MemeEncoder, build_point, deterministic_id

UPSERT_BATCH = 32
DISCOVER_EXTENSIONS = PHOTO_EXTENSIONS
LOCK_PATH = Path(__file__).resolve().parent / ".lumina.lock"
STATE_VERSION = 2
SWEEP_INTERVAL_SECONDS = 3600


def discover_images(folder: Path, recursive: bool = True) -> list[str]:
    """List image files under ``folder``, skipping dotfiles and hidden dirs."""
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    paths = []
    for path in iterator:
        if not path.is_file() or path.suffix.lower() not in DISCOVER_EXTENSIONS:
            continue
        relative_parts = path.relative_to(folder).parts
        if any(part.startswith(".") for part in relative_parts):
            continue  # .Trashes, .Spotlight-V100, hidden folders, etc.
        paths.append(str(path))
    return sorted(paths)


def content_digest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stat_signature(stat_result: os.stat_result) -> str:
    """Cheap identity probe: identical signature implies identical bytes."""
    return f"{stat_result.st_size}:{stat_result.st_mtime_ns}"


class InstanceLock:
    """Advisory single-instance guard; embedded Qdrant allows one writer."""

    def __init__(self, path: Path = LOCK_PATH) -> None:
        self.path = path
        self.acquired = False

    def __enter__(self) -> "InstanceLock":
        if self.path.exists():
            try:
                pid = int(self.path.read_text().strip())
                os.kill(pid, 0)
                raise SystemExit(
                    f"Another Lumina ingestion is running (pid {pid}). "
                    "If that is wrong, delete the stale lock: "
                    f"rm '{self.path}'"
                )
            except ProcessLookupError:
                self.path.unlink(missing_ok=True)  # stale lock from a dead run
        self.path.write_text(str(os.getpid()))
        self.acquired = True
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)


def load_state(state_path: Path) -> dict[str, Any]:
    try:
        state = json.loads(state_path.read_text())
    except (OSError, ValueError):
        return {"version": STATE_VERSION}
    # v1 states carried no per-file cache; start fresh rather than trusting it.
    if state.get("version") != STATE_VERSION:
        state = {"version": STATE_VERSION}
    return state


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    state["version"] = STATE_VERSION
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    state_path.write_text(json.dumps(state, indent=2))


def repair_moved_file(client: Any, collection_name: str, point_id: str, new_path: str, stored_path: str) -> bool:
    """Repoint an existing point at its new location without re-indexing."""
    try:
        client.set_payload(
            collection_name=collection_name,
            payload={"local_path": new_path},
            points=[point_id],
            wait=True,
        )
        print(f"Repaired moved file: {Path(stored_path).name} -> {new_path}")
        return True
    except Exception as exc:
        print(f"Could not repoint {new_path}: {exc}")
        return False


def sweep_stale_points(client: Any, collection_name: str) -> int:
    """Delete points whose file vanished or whose content was replaced.

    A point is stale when ``local_path`` no longer exists, or when the file
    there now hashes differently (edited/re-saved image superseded it — the
    new version is indexed as its own content-hash point).
    """
    stale_ids: list[str] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            path = str(payload.get("local_path", ""))
            content_hash = str(payload.get("content_hash", ""))
            if not path:
                continue
            if not os.path.exists(path):
                stale_ids.append(str(point.id))
                print(f"Sweeping missing file: {path}")
            elif content_hash:
                try:
                    if content_digest(path) != content_hash:
                        stale_ids.append(str(point.id))
                        print(f"Sweeping superseded version: {path}")
                except OSError:
                    pass  # unreadable right now; assume it is fine
        if offset is None:
            break
    for start in range(0, len(stale_ids), 256):
        client.delete(
            collection_name=collection_name,
            points_selector=PointIdsList(points=stale_ids[start : start + 256]),
            wait=True,
        )
    return len(stale_ids)


def scan_library(
    folder: Path,
    collection_name: str,
    *,
    encoder: MemeEncoder,
    captioner: Any,
    client: Any,
    state: dict[str, Any],
    state_path: Path | None = None,
    limit: int = 0,
    recursive: bool = True,
    do_sweep: bool = True,
) -> dict[str, int]:
    """Run one incremental pass over ``folder`` and merge results into state.

    Unchanged files are skipped via cached stat signatures without hashing;
    anything genuinely new/changed/failed goes through content-hash dedup,
    move-repair, VLM captioning, and embedding exactly once.
    """
    paths = discover_images(folder, recursive=recursive)
    if limit:
        paths = paths[:limit]
    print(f"Discovered {len(paths)} images under {folder}")

    stats = {"indexed": 0, "unchanged": 0, "repaired": 0, "failed": 0}
    failures: dict[str, str] = {}
    file_cache: dict[str, Any] = state.setdefault("files", {})
    pending: list[tuple[str, str, str]] = []  # (path, digest, stat signature)

    def remember(path: str, digest: str, signature: str, status: str) -> None:
        file_cache[path] = {"sig": signature, "digest": digest, "status": status}

    def flush(pending: list[tuple[str, str, str]]) -> None:
        if not pending:
            return
        ids = [deterministic_id(digest) for _, digest, _ in pending]
        records = client.retrieve(collection_name=collection_name, ids=ids, with_payload=True)
        by_id = {str(record.id): record for record in records}
        buffer: list[Any] = []
        for (path, digest, signature), point_id in zip(pending, ids):
            record = by_id.get(point_id)
            if record is not None:
                stored_path = str((record.payload or {}).get("local_path", ""))
                if stored_path == path:
                    stats["unchanged"] += 1
                elif repair_moved_file(client, collection_name, point_id, path, stored_path):
                    stats["repaired"] += 1
                remember(path, digest, signature, "indexed")  # confirmed present either way
                continue
            try:
                media_type = detect_media_type(path)
                captured_at = capture_datetime(path)
                caption_text = ""
                is_sensitive = False
                if captioner is not None:
                    result = captioner.caption_image(open_image(path), media_type=media_type)
                    caption_text, is_sensitive = result.caption, result.is_sensitive
                buffer.append(
                    build_point(
                        path,
                        encoder,
                        tags=(media_type,),
                        caption=caption_text,
                        media_type=media_type,
                        captured_at=captured_at,
                        metadata={"is_sensitive": is_sensitive},
                    )
                )
            except Exception as exc:
                stats["failed"] += 1
                failures[path] = str(exc)[:300]
                remember(path, digest, signature, "failed")
                print(f"Skipping {Path(path).name}: {exc}")
        if buffer:
            client.upsert(collection_name=collection_name, points=buffer, wait=True)
            stats["indexed"] += len(buffer)
        # Cache every settled outcome (indexed or already-present) with its
        # real stat signature so future scans skip straight past them.
        # Failures keep their status so they retry next scan.
        for path, digest, signature in pending:
            if path in failures:
                continue
            remember(path, digest, signature, "indexed")

    total_work = max(len(paths), 1)
    processed = 0
    for path in paths:
        try:
            signature = stat_signature(os.stat(path))
        except OSError as exc:
            stats["failed"] += 1
            failures[path] = str(exc)[:300]
            print(f"Skipping {path}: {exc}")
            continue
        cached = file_cache.get(path)
        if os.environ.get("LUMINA_DEBUG"):
            print(f"DEBUG {Path(path).name}: cached={'yes' if cached else 'no'} "
                  f"cached_sig={cached.get('sig') if cached else None} current={signature}",
                  file=sys.stderr)
        if cached and cached.get("status") == "indexed" and cached.get("sig") == signature:
            stats["unchanged"] += 1  # no hash, no Qdrant round-trip
            continue
        try:
            pending.append((path, content_digest(path), signature))
        except OSError as exc:
            stats["failed"] += 1
            failures[path] = str(exc)[:300]
            print(f"Skipping {path}: {exc}")
            continue
        if len(pending) >= UPSERT_BATCH:
            flush(pending)
            pending.clear()
            processed += 1
            _snapshot_state(state, folder, stats, failures, len(paths), f"{processed}/{total_work}")
            if state_path:
                save_state(state_path, state)

    # Flush whatever remains — cache hits on trailing files used to strand
    # this buffer when the flush was keyed to the loop index instead.
    if pending:
        flush(pending)
        pending.clear()

    # Prune cache entries for files that left the library (full scans only —
    # a --limit scan sees a truncated view and must not evict real entries).
    if not limit:
        discovered = set(paths)
        for cached_path in list(file_cache):
            if cached_path not in discovered and cached_path.startswith(str(folder)):
                del file_cache[cached_path]

    _snapshot_state(state, folder, stats, failures, len(paths))
    if state_path:
        save_state(state_path, state)

    # Sweep after ingestion: surviving points were already repointed, so only
    # genuinely dead/superseded points die here.
    if do_sweep:
        swept = sweep_stale_points(client, collection_name)
        if swept:
            print(f"Swept {swept} stale points")

    print(
        f"Scan done · indexed {stats['indexed']} · unchanged {stats['unchanged']} · "
        f"moved+repaired {stats['repaired']} · failed {stats['failed']}"
    )
    return stats


def _snapshot_state(
    state: dict[str, Any],
    folder: Path,
    stats: dict[str, int],
    failures: dict[str, str],
    discovered: int,
    progress: str | None = None,
) -> None:
    last_scan: dict[str, Any] = {**stats, "discovered": discovered}
    if progress:
        last_scan["progress"] = progress
    state["last_scan"] = last_scan
    state["folders"] = [str(folder)]
    state["failures"] = failures


def load_models() -> tuple[MemeEncoder, Any]:
    encoder = MemeEncoder()
    try:
        from captioner import build_captioner

        captioner = build_captioner()
    except Exception as exc:
        print(f"Captioning unavailable ({exc}); continuing with OCR + visual search only")
        captioner = None
    return encoder, captioner


def watch_forever(
    folder: Path,
    collection_name: str,
    *,
    debounce_seconds: float,
    recursive: bool = True,
) -> None:
    """Initial scan, then rescan after filesystem activity settles."""
    from watcher import LibraryWatcher

    encoder, captioner = load_models()
    client = qdrant_client()
    state_path = Path(MEMORY_STATE_PATH)
    state = load_state(state_path)
    last_sweep = [0.0]

    def scan_now(trigger: str) -> None:
        due_sweep = (time.time() - last_sweep[0]) >= SWEEP_INTERVAL_SECONDS
        try:
            scan_library(
                folder,
                collection_name,
                encoder=encoder,
                captioner=captioner,
                client=client,
                state=state,
                state_path=state_path,
                recursive=recursive,
                do_sweep=due_sweep,
            )
            if due_sweep:
                last_sweep[0] = time.time()
        except Exception as exc:
            # Embedded storage can be briefly locked right after startup;
            # never let one bad cycle kill the watcher.
            print(f"WARNING: watch scan failed ({trigger}): {exc}; will retry on next change")

    print(
        f"Watching {folder} — indexing changes after {debounce_seconds:.0f}s of quiet. "
        "Ctrl+C to stop."
    )
    scan_now("startup")
    watcher = LibraryWatcher([str(folder)], scan_now, debounce_seconds=debounce_seconds)
    watcher.start()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        watcher.stop()
        print("Watcher stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", default=SCREENSHOT_ROOT, help="folder to index (default ~/Screenshots)")
    parser.add_argument("--collection", default=MEMORY_COLLECTION_NAME)
    parser.add_argument("--limit", type=int, default=0, help="cap indexed files (0 = all)")
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--no-sweep", action="store_true", help="skip stale-point cleanup")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="stay running and rescan whenever the folder changes",
    )
    parser.add_argument("--debounce", type=float, default=10.0, help="seconds of quiet before a watch rescan")
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"Not a folder: {folder}")

    initialize_meme_database(args.collection)

    with InstanceLock():
        if args.watch:
            watch_forever(
                folder,
                args.collection,
                debounce_seconds=max(args.debounce, 1.0),
                recursive=not args.no_recursive,
            )
            return
        encoder, captioner = load_models()
        client = qdrant_client()
        state_path = Path(MEMORY_STATE_PATH)
        state = load_state(state_path)
        scan_library(
            folder,
            args.collection,
            encoder=encoder,
            captioner=captioner,
            client=client,
            state=state,
            state_path=state_path,
            limit=args.limit,
            recursive=not args.no_recursive,
            do_sweep=not args.no_sweep,
        )


if __name__ == "__main__":
    main()
