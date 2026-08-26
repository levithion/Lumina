# Watch mode — keeping the index fresh

Lumina can stay running and re-index your library automatically whenever
files change. Nothing leaves your machine: FSEvents notices the change,
Lumina waits for a quiet moment, then runs an incremental scan.

## One-shot vs watch

```bash
# index once and exit
python ingest_screenshots.py

# keep running; rescan ~10s after filesystem activity settles
python ingest_screenshots.py --watch
```

Useful flags:

| Flag | Meaning |
| --- | --- |
| `--folder PATH` | library root (default `~/Screenshots`) |
| `--debounce SECONDS` | quiet period before a rescan (default 10, min 1) |
| `--no-sweep` | skip stale-point cleanup for this run |
| `LUMINA_DEBUG=1` | print per-file cache decisions to stderr |

## How freshness works

- **Stat-signature cache** (`.lumina_state.json`): unchanged files are
  detected by size + mtime alone — no re-hashing, no model calls. This is
  what makes frequent rescans cheap even with tens of thousands of files.
- **Content-hash identity**: anything new/changed is hashed once; identical
  content in a new location repairs the existing point instead of
  re-captioning.
- **Debounce**: saving one screenshot fires many FS events; scans trigger
  only after the configured quiet window.
- **Sweep throttle**: stale-point cleanup (deleted/superseded files) runs on
  startup and at most hourly during watch, since it walks the whole index.

## Single-process rule

Embedded Qdrant (`qdrant_data/`) allows exactly one process per data
directory. The intended setup is `python main.py`, which hosts models,
watcher, API, and database **in one process** — no conflict is possible.

Standalone alternatives exist but must not run alongside `main.py`:

- `python ingest_screenshots.py --watch` — watcher only
- `python server.py` / `--web` — API only

If you start a second process against the same store you'll be told another
instance holds it; stop one or point `QDRANT_URL` at a Qdrant server.

## Survive reboots (launchd)

See [`launchd/com.lumina.watch.plist`](../launchd/com.lumina.watch.plist) —
edit your username/project path into it, then:

```bash
cp launchd/com.lumina.watch.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.lumina.watch.plist
```

The plist runs `main.py --web` at login (API + watcher, no window). Logs land
in `/tmp/lumina-watch.log`. Unload + delete the plist to remove.

## State file

`.lumina_state.json` (project root, gitignored) records per-file digests,
scan stats, and failures. Failed files retry on the next scan; deleting the
file simply makes the next run re-check everything from scratch (slower,
never wrong).
