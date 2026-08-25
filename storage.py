"""Optional public Hugging Face Dataset storage for indexed meme images."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from config import HF_DATASET_REPO, HF_TOKEN

# Files per git commit; huge single commits are slow and can time out.
COMMIT_BATCH = 40


def enabled() -> bool:
    return bool(HF_TOKEN and HF_DATASET_REPO)


def _resolve_url(object_name: str) -> str:
    owner, name = HF_DATASET_REPO.split("/", 1)
    base = f"https://huggingface.co/datasets/{quote(owner)}/{quote(name)}/resolve/main"
    return f"{base}/{quote(object_name, safe='/')}"


def upload_images(pairs: list[tuple[str, str]]) -> dict[str, str]:
    """Upload many local files in batched commits.

    ``pairs`` maps local paths to repository paths ("memes/<sha>.png").
    Returns ``{repo_path: public resolve URL}``.
    """
    if not pairs:
        return {}
    if not enabled():
        raise RuntimeError("Set HF_TOKEN and HF_DATASET_REPO before uploading")
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_DATASET_REPO, repo_type="dataset", private=False, exist_ok=True)
    urls: dict[str, str] = {}
    for start in range(0, len(pairs), COMMIT_BATCH):
        chunk = pairs[start : start + COMMIT_BATCH]
        operations = [
            CommitOperationAdd(path_or_fileobj=str(local_path), path_in_repo=object_name)
            for local_path, object_name in chunk
        ]
        api.create_commit(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            operations=operations,
            # Required keyword in huggingface_hub 1.x; omitting it raises.
            commit_message=f"Add {len(chunk)} meme image(s)",
            token=HF_TOKEN,
        )
        urls.update({object_name: _resolve_url(object_name) for _, object_name in chunk})
    return urls


def upload_image(local_path: str, object_name: str | None = None) -> str:
    """Single-file convenience wrapper around :func:`upload_images`."""
    repo_path = object_name or f"memes/{Path(local_path).name}"
    return upload_images([(local_path, repo_path)]).get(repo_path, "")
