"""Launch Lumina as a native-feeling desktop app.

Boots the API server (models + embedded Qdrant + optional watcher) in a
background thread, then opens a native macOS window via pywebview/WebKit.

    python main.py                 # windowed app, watching ~/Screenshots
    python main.py --no-watch      # window without filesystem watcher
    python main.py --web           # headless: print URL instead of a window
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import urllib.request
from pathlib import Path

import uvicorn

from config import SCREENSHOT_ROOT, SERVER_HOST, SERVER_PORT
from server import create_app


def wait_until_ready(port: int, timeout_seconds: float = 300) -> bool:
    """Block until /api/status answers — models may take a minute to load."""
    deadline = time.time() + timeout_seconds
    url = f"http://127.0.0.1:{port}/api/status"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=SERVER_HOST)
    parser.add_argument("--port", type=int, default=SERVER_PORT)
    parser.add_argument("--folder", default=SCREENSHOT_ROOT)
    parser.add_argument("--collection", default="")
    parser.add_argument("--no-watch", action="store_true", help="disable live folder watching")
    parser.add_argument("--web", action="store_true", help="serve headlessly; skip the native window")
    args = parser.parse_args()

    app = create_app(
        folder=Path(args.folder).expanduser().resolve(),
        watch=not args.no_watch,
        **({"collection": args.collection} if args.collection else {}),
    )
    server = uvicorn.Server(
        uvicorn.Config(app, host=args.host, port=args.port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, name="lumina-server", daemon=True)
    thread.start()

    print("Loading Lumina (first start indexes models)…")
    if not wait_until_ready(args.port):
        print("Server failed to become ready; see logs above.", file=sys.stderr)
        sys.exit(1)

    url = f"http://{args.host}:{args.port}/"
    if args.web:
        print(f"Lumina ready at {url} — Ctrl+C to stop.")
        try:
            while thread.is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        return

    try:
        import webview
    except ImportError:
        print("pywebview missing — falling back to web mode.")
        print(f"Lumina ready at {url}")
        while thread.is_alive():
            time.sleep(1)
        return

    webview.create_window(
        "Lumina",
        url,
        width=1320,
        height=860,
        min_size=(980, 620),
        background_color="#0e1116",
    )
    webview.start()  # blocks until the window closes; daemon threads exit with it


if __name__ == "__main__":
    main()
