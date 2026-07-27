"""Official source adapters for fresh meme ingestion."""

from __future__ import annotations

from typing import Any

import requests

from config import (
    LIVE_FETCH_LIMIT,
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_PASSWORD,
    REDDIT_SUBREDDITS,
    REDDIT_USER_AGENT,
    REDDIT_USERNAME,
    IMGUR_CLIENT_ID,
    IMGUR_GALLERY_PATH,
    IMGUR_KEYWORDS,
    IMGUR_MIN_SCORE,
    MEME_API_SUBREDDITS,
)


def fetch_imgflip(limit: int = LIVE_FETCH_LIMIT) -> list[dict[str, Any]]:
    response = requests.get("https://api.imgflip.com/get_memes", timeout=20)
    response.raise_for_status()
    memes = response.json().get("data", {}).get("memes", [])
    return [
        {
            "source": "imgflip",
            "source_id": str(item.get("id")),
            "source_url": item.get("url", ""),
            "image_url": item.get("url", ""),
            "title": item.get("name", ""),
            "template": item.get("name", ""),
            "tags": ["template", "popular"],
        }
        for item in memes[:limit]
        if item.get("url")
    ]


def fetch_imgur(limit: int = LIVE_FETCH_LIMIT) -> list[dict[str, Any]]:
    """Fetch actual public gallery images, not blank meme templates."""
    if not IMGUR_CLIENT_ID:
        raise RuntimeError("Set IMGUR_CLIENT_ID before using the Imgur source")
    response = requests.get(
        f"https://api.imgur.com/{IMGUR_GALLERY_PATH.lstrip('/')}",
        params={"showViral": "true", "mature": "false", "album_previews": "false"},
        headers={"Authorization": f"Client-ID {IMGUR_CLIENT_ID}"},
        timeout=20,
    )
    response.raise_for_status()
    keywords = {word.strip().lower() for word in IMGUR_KEYWORDS.split(",") if word.strip()}
    items: list[dict[str, Any]] = []
    for post in response.json().get("data", []):
        if post.get("is_album") or not post.get("link") or int(post.get("points") or 0) < IMGUR_MIN_SCORE:
            continue
        haystack = " ".join([
            str(post.get("title") or ""),
            str(post.get("description") or ""),
            str(post.get("topic") or ""),
            " ".join(str(tag) for tag in (post.get("tags") or [])),
        ]).lower()
        if keywords and not any(keyword in haystack for keyword in keywords):
            continue
        items.append({
            "source": "imgur",
            "source_id": str(post.get("id", "")),
            "source_url": f"https://imgur.com/{post.get('id', '')}",
            "image_url": post.get("link", ""),
            "title": post.get("title", ""),
            "template": "",
            "tags": ["imgur", "gallery"],
            "created_at": post.get("datetime"),
            "score": post.get("points", 0),
        })
        if len(items) >= limit:
            break
    return items


def fetch_meme_api(limit: int = LIVE_FETCH_LIMIT) -> list[dict[str, Any]]:
    """Fetch actual captioned memes from meme-api.com's Reddit-backed feed.

    This is a temporary fallback while official Reddit access is pending; it
    must not be treated as a replacement for an approved Reddit integration.
    """
    names = [name.strip() for name in MEME_API_SUBREDDITS.split(",") if name.strip()]
    per_subreddit = max(1, min(50, limit // max(len(names), 1)))
    items: list[dict[str, Any]] = []
    for subreddit in names:
        response = requests.get(
            f"https://meme-api.com/gimme/{subreddit}/{per_subreddit}",
            headers={"User-Agent": REDDIT_USER_AGENT},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        posts = payload.get("memes", [payload] if payload.get("url") else [])
        for post in posts:
            url = post.get("url", "")
            if post.get("nsfw") or not url.lower().split("?")[0].endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")):
                continue
            items.append({
                "source": "meme-api",
                "source_id": post.get("postLink", ""),
                "source_url": post.get("postLink", ""),
                "image_url": url,
                "title": post.get("title", ""),
                "subreddit": post.get("subreddit", subreddit),
                "template": "",
                "tags": [post.get("subreddit", subreddit), "reddit"],
                "created_at": post.get("created", ""),
                "score": post.get("ups", 0),
            })
    return items[:limit]


def _reddit_token() -> str:
    if not all((REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)):
        raise RuntimeError("Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET")
    grant = {"grant_type": "client_credentials"}
    # Password grant is retained for apps whose Reddit registration requires a
    # user context; app-only credentials are preferable for a read-only feed.
    if REDDIT_USERNAME and REDDIT_PASSWORD:
        grant = {"grant_type": "password", "username": REDDIT_USERNAME, "password": REDDIT_PASSWORD}
    response = requests.post(
        "https://www.reddit.com/api/v1/access_token",
        auth=(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET),
        data=grant,
        headers={"User-Agent": REDDIT_USER_AGENT},
        timeout=20,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def fetch_reddit(subreddits: list[str] | None = None, limit: int = LIVE_FETCH_LIMIT) -> list[dict[str, Any]]:
    token = _reddit_token()
    headers = {"Authorization": f"Bearer {token}", "User-Agent": REDDIT_USER_AGENT}
    names = subreddits or [name.strip() for name in REDDIT_SUBREDDITS.split(",") if name.strip()]
    items: list[dict[str, Any]] = []
    for subreddit in names:
        response = requests.get(
            f"https://oauth.reddit.com/r/{subreddit}/new",
            params={"limit": min(limit, 100), "raw_json": 1},
            headers=headers,
            timeout=20,
        )
        response.raise_for_status()
        for child in response.json().get("data", {}).get("children", []):
            post = child.get("data", {})
            url = post.get("url_overridden_by_dest") or post.get("url", "")
            if post.get("is_video") or not url.lower().split("?")[0].endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")):
                continue
            items.append({
                "source": "reddit",
                "source_id": post.get("id", ""),
                "source_url": f"https://www.reddit.com{post.get('permalink', '')}",
                "image_url": url,
                "title": post.get("title", ""),
                "subreddit": post.get("subreddit", subreddit),
                "template": "",
                "tags": [subreddit, "reddit"],
                "created_at": post.get("created_utc"),
                "score": post.get("score", 0),
            })
    return items
