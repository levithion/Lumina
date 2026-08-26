"""Filesystem watcher with debounce for keeping the Lumina index fresh.

FSEvents fires bursts of events for a single save; ``LibraryWatcher`` waits
for quiet before triggering exactly one rescan. The callback runs on a timer
thread and must be safe to re-enter (``scan_library`` is idempotent).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer


class _Debouncer:
    """Collapses rapid notifications into one trailing-edge callback."""

    def __init__(self, callback: Callable[[str], None], delay: float) -> None:
        self._callback = callback
        self._delay = max(delay, 0.05)
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def notify(self, trigger: str) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            timer = threading.Timer(self._delay, self._fire)
            timer.daemon = True
            self._timer = timer
        timer.start()

    def _fire(self) -> None:
        with self._lock:
            self._timer = None
        try:
            self._callback("changes detected")
        except Exception as exc:  # never let the watcher thread die
            print(f"WARNING: watch callback failed: {exc}")

    def cancel(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None


class _EventHandler(FileSystemEventHandler):
    def __init__(self, debouncer: _Debouncer) -> None:
        super().__init__()
        self._debouncer = debouncer

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return  # directory churn alone carries no index work
        self._debouncer.notify(str(getattr(event, "src_path", "")))


class LibraryWatcher:
    """Watch one or more folders; call ``callback`` once things go quiet."""

    def __init__(
        self,
        folders: list[str],
        callback: Callable[[str], None],
        *,
        debounce_seconds: float = 10.0,
    ) -> None:
        self.folders = [str(Path(folder).resolve()) for folder in folders]
        self.callback = callback
        self.debounce_seconds = debounce_seconds
        self._observer: Any = None
        self._debouncer: _Debouncer | None = None

    @property
    def is_running(self) -> bool:
        return self._observer is not None and self._observer.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._debouncer = _Debouncer(self.callback, self.debounce_seconds)
        observer = Observer(timeout=1.0)
        handler = _EventHandler(self._debouncer)
        for folder in self.folders:
            observer.schedule(handler, folder, recursive=True)
        observer.daemon_threads = True
        observer.start()
        self._observer = observer

    def stop(self) -> None:
        if self._debouncer is not None:
            self._debouncer.cancel()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
