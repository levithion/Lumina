"""Media helpers for private memory search: HEIC, EXIF dates, media typing."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image

HEIC_EXTENSIONS = {".heic", ".heif"}
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"} | HEIC_EXTENSIONS

_HEIF_REGISTERED = False

# EXIF tags that carry capture time, in priority order. SubSecond refines
# nothing for our day-level use, so it is ignored.
_EXIF_DATETIME_TAGS = (0x0132,)  # DateTime
_EXIF_DATETIME_ORIGINAL_TAG = 0x9003  # DateTimeOriginal
_EXIF_DIGITIZED_TAG = 0x9004  # DateTimeDigitized


def ensure_heif_support() -> bool:
    """Enable Pillow's HEIC/HEIF decoder when pillow-heif is installed.

    Safe to call repeatedly; returns whether support is active.
    """
    global _HEIF_REGISTERED
    if _HEIF_REGISTERED:
        return True
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
        _HEIF_REGISTERED = True
        return True
    except ImportError:
        return False


def open_image(path: str) -> Image.Image:
    """Open an image as RGB, registering HEIC support on first use."""
    if Path(path).suffix.lower() in HEIC_EXTENSIONS:
        if not ensure_heif_support():
            raise RuntimeError(
                f"{Path(path).name}: .heic file but pillow-heif is not installed "
                "(add it to requirements and reinstall)"
            )
    return Image.open(path).convert("RGB")


def detect_media_type(path: str) -> str:
    """Best-effort split between screenshots and camera photos.

    macOS screenshots follow a "Screenshot 2026-08-25 at 09.41.05.png" naming
    scheme; iOS files use IMG_/Screenshot prefixes. Everything else that is a
    photo extension counts as a photo.
    """
    name = Path(path).name.lower()
    if name.startswith("screenshot") or name.startswith("screen shot"):
        return "screenshot"
    if "_screen" in name and name.endswith((".png", ".jpg", ".jpeg")):
        return "screenshot"
    return "photo"


def exif_capture_datetime(path: str) -> datetime | None:
    """Capture time from EXIF, preferring DateTimeOriginal over digitized.

    Returns timezone-aware datetimes: local offset when EXIF carries one,
    otherwise naive local time interpreted in the machine's zone.
    """
    try:
        with open_image(path) as image:
            exif = image.getexif()
    except Exception:
        return None

    def _parse(raw_tag_value: object) -> datetime | None:
        if not raw_tag_value:
            return None
        try:
            parsed = datetime.strptime(str(raw_tag_value).strip(), "%Y:%m:%d %H:%M:%S")
        except ValueError:
            return None
        return parsed.astimezone()

    candidates = (
        _parse(exif.get(_EXIF_DATETIME_ORIGINAL_TAG)),
        _parse(exif.get(_EXIF_DIGITIZED_TAG)),
        _parse(exif.get(_EXIF_DATETIME_TAGS[0])),
    )
    return next((value for value in candidates if value is not None), None)


def capture_datetime(path: str) -> str:
    """ISO capture time for a media file: EXIF first, file mtime fallback."""
    from_exif = exif_capture_datetime(path)
    if from_exif is not None:
        return from_exif.isoformat()
    mtime = os.stat(path).st_mtime
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
